use crate::data_provider::DataProvider;
use crate::executor::{EvalExecutor, Executor};
use crate::expr::{Expr, Literal};
use crate::logical_plan::LogicalPlan;
use crate::optimizer::Optimizer;
use ahash::AHasher;
use bincode;
use lru::LruCache;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hasher;
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Serialize, Deserialize)]
pub struct CacheDump {
    pub version: u32,
    pub items: Vec<(String, f64)>,
}

#[derive(Clone, Debug)]
pub struct Chromosome {
    pub genes: Vec<u8>,
}

pub trait GenotypeDecoder: Sync {
    fn decode(&self, c: &Chromosome) -> Expr;
}
pub trait FitnessEvaluator: Send + Sync {
    fn fitness(&self, expr: &Expr) -> f64;
}

/// Evaluate population in parallel using rayon
pub fn evaluate_population(
    decoder: &dyn GenotypeDecoder,
    evaluator: &dyn FitnessEvaluator,
    pop: &[Chromosome],
) -> Vec<f64> {
    pop.par_iter()
        .map(|c| {
            let e = decoder.decode(c);
            if !is_executable_expr(&e) {
                return -1e9;
            }
            evaluator.fitness(&e)
        })
        .collect()
}

/// Single-point crossover for vector-encoded chromosomes
pub fn crossover(a: &Chromosome, b: &Chromosome, rng: &mut impl Rng) -> (Chromosome, Chromosome) {
    let len = a.genes.len().min(b.genes.len());
    if len == 0 {
        return (a.clone(), b.clone());
    }
    let point = rng.gen_range(0..len);
    let mut c1 = a.genes.clone();
    let mut c2 = b.genes.clone();
    for i in point..len {
        c1[i] = b.genes[i];
        c2[i] = a.genes[i];
    }
    (Chromosome { genes: c1 }, Chromosome { genes: c2 })
}

/// Mutate chromosome in-place: each gene has `rate` chance to be replaced by a random value from 0..max_val
pub fn mutate(ch: &mut Chromosome, rate: f64, max_val: u8, rng: &mut impl Rng) {
    for g in ch.genes.iter_mut() {
        if rng.gen_bool(rate) {
            *g = rng.gen_range(0..max_val);
        }
    }
}

fn random_chromosome(len: usize, max_val: u8, rng: &mut impl Rng) -> Chromosome {
    let mut genes = Vec::with_capacity(len);
    for _ in 0..len {
        genes.push(rng.gen_range(0..max_val));
    }
    Chromosome { genes }
}

pub struct GAConfig {
    pub generations: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub gene_max: u8,
}

/// Run a simple generational GA. `initial` is used as starting population.
/// Returns best chromosome and its fitness.
pub fn run_ga(
    config: &GAConfig,
    mut population: Vec<Chromosome>,
    decoder: &dyn GenotypeDecoder,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut impl Rng,
) -> (Chromosome, f64) {
    let pop_size = population.len();
    assert!(pop_size > 0, "population must be non-empty");

    // evaluate initial
    // ensure initial population contains only executable expressions; replace invalid with random chromosomes
    for ch in population.iter_mut() {
        let e = decoder.decode(ch);
        if !is_executable_expr(&e) {
            // replace with random chromosome of same length
            *ch = random_chromosome(ch.genes.len(), config.gene_max, rng);
        }
    }
    let mut scores = evaluate_population(decoder, evaluator, &population);

    for _gen in 0..config.generations {
        // selection + reproduce
        let mut new_pop: Vec<Chromosome> = Vec::with_capacity(pop_size);
        while new_pop.len() < pop_size {
            // tournament select parent 1
            let mut best_idx: usize = rng.gen_range(0usize..pop_size);
            for _ in 1..config.tournament_size {
                let idx: usize = rng.gen_range(0usize..pop_size);
                if scores[idx] > scores[best_idx] {
                    best_idx = idx;
                }
            }
            let p1 = population[best_idx].clone();

            // tournament select parent 2
            let mut best_idx2: usize = rng.gen_range(0usize..pop_size);
            for _ in 1..config.tournament_size {
                let idx: usize = rng.gen_range(0usize..pop_size);
                if scores[idx] > scores[best_idx2] {
                    best_idx2 = idx;
                }
            }
            let p2 = population[best_idx2].clone();

            if rng.gen_bool(config.crossover_rate) {
                let (c1, c2) = crossover(&p1, &p2, rng);
                new_pop.push(c1);
                if new_pop.len() < pop_size {
                    new_pop.push(c2);
                }
            } else {
                new_pop.push(p1);
                if new_pop.len() < pop_size {
                    new_pop.push(p2);
                }
            }
        }

        // mutation
        for ch in new_pop.iter_mut() {
            mutate(ch, config.mutation_rate, config.gene_max, rng);
        }

        // replace invalid decoded exprs in new_pop
        for ch in new_pop.iter_mut() {
            let e = decoder.decode(ch);
            if !is_executable_expr(&e) {
                *ch = random_chromosome(ch.genes.len(), config.gene_max, rng);
            }
        }
        // evaluate new population
        scores = evaluate_population(decoder, evaluator, &new_pop);
        population = new_pop;
    }

    // return best
    let mut best_idx = 0usize;
    for i in 1..population.len() {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }
    (population[best_idx].clone(), scores[best_idx])
}

/// Basic check whether Expr contains unsupported nodes (e.g., Func)
fn is_executable_expr(e: &Expr) -> bool {
    match e {
        Expr::Func { .. } => false,
        Expr::UnaryOp { input, .. } => is_executable_expr(input),
        Expr::BinaryOp { left, right, .. } => is_executable_expr(left) && is_executable_expr(right),
        _ => true,
    }
}

// --- Tree-based GA utilities ---

/// Collect all paths to nodes in the tree. A path is a sequence of child indices as described:
/// - Unary: child 0
/// - Binary: left 0, right 1
/// - Func: args indexed 0..n-1
fn collect_paths(e: &Expr, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    out.push(cur.clone());
    match e {
        Expr::UnaryOp { input, .. } => {
            cur.push(0);
            collect_paths(input, cur, out);
            cur.pop();
        }
        Expr::BinaryOp { left, right, .. } => {
            cur.push(0);
            collect_paths(left, cur, out);
            cur.pop();
            cur.push(1);
            collect_paths(right, cur, out);
            cur.pop();
        }
        Expr::Func { args, .. } => {
            for i in 0..args.len() {
                cur.push(i);
                collect_paths(&args[i], cur, out);
                cur.pop();
            }
        }
        _ => {}
    }
}

fn get_subtree(e: &Expr, path: &[usize]) -> Expr {
    if path.is_empty() {
        return e.clone();
    }
    let mut node = e;
    for &idx in path {
        node = match node {
            Expr::UnaryOp { input, .. } => &*input,
            Expr::BinaryOp { left, right, .. } => {
                if idx == 0 {
                    &*left
                } else {
                    &*right
                }
            }
            Expr::Func { args, .. } => &args[idx],
            _ => {
                return node.clone();
            }
        };
    }
    node.clone()
}

fn replace_subtree(e: &mut Expr, path: &[usize], new_sub: Expr) {
    if path.is_empty() {
        *e = new_sub;
        return;
    }
    let mut node = e;
    for (i, &idx) in path.iter().enumerate() {
        if i == path.len() - 1 {
            match node {
                Expr::UnaryOp { input, .. } => {
                    *input = Box::new(new_sub);
                    return;
                }
                Expr::BinaryOp { left, right, .. } => {
                    if idx == 0 {
                        *left = Box::new(new_sub);
                    } else {
                        *right = Box::new(new_sub);
                    }
                    return;
                }
                Expr::Func { args, .. } => {
                    args[idx] = new_sub;
                    return;
                }
                _ => {
                    *node = new_sub;
                    return;
                }
            }
        } else {
            node = match node {
                Expr::UnaryOp { input, .. } => &mut *input,
                Expr::BinaryOp { left, right, .. } => {
                    if idx == 0 {
                        &mut *left
                    } else {
                        &mut *right
                    }
                }
                Expr::Func { args, .. } => &mut args[idx],
                _ => return,
            };
        }
    }
}

/// Subtree crossover: pick random node path in each tree and swap the subtrees.
pub fn subtree_crossover(a: &Expr, b: &Expr, rng: &mut impl Rng) -> (Expr, Expr) {
    let mut paths_a = Vec::new();
    collect_paths(a, &mut Vec::new(), &mut paths_a);
    let mut paths_b = Vec::new();
    collect_paths(b, &mut Vec::new(), &mut paths_b);
    if paths_a.is_empty() || paths_b.is_empty() {
        return (a.clone(), b.clone());
    }
    let pa = &paths_a[rng.gen_range(0..paths_a.len())];
    let pb = &paths_b[rng.gen_range(0..paths_b.len())];
    let sub_a = get_subtree(a, pa);
    let sub_b = get_subtree(b, pb);
    let mut na = a.clone();
    let mut nb = b.clone();
    replace_subtree(&mut na, pa, sub_b);
    replace_subtree(&mut nb, pb, sub_a);
    (na, nb)
}

/// Subtree mutation: replace a random subtree with a new random small subtree.
pub fn subtree_mutate(e: &Expr, max_depth: usize, rng: &mut impl Rng) -> Expr {
    let mut paths = Vec::new();
    collect_paths(e, &mut Vec::new(), &mut paths);
    if paths.is_empty() {
        return e.clone();
    }
    let p = &paths[rng.gen_range(0..paths.len())];
    let mut ne = e.clone();
    let new_sub = random_expr(rng, max_depth);
    replace_subtree(&mut ne, p, new_sub);
    ne
}

fn random_expr(rng: &mut impl Rng, max_depth: usize) -> Expr {
    if max_depth == 0 {
        return Expr::Lit(rng.gen_range(0.0..10.0));
    }
    let choice = rng.gen_range(0..4);
    match choice {
        0 => Expr::Lit(rng.gen_range(0.0..10.0)),
        1 => Expr::Col("x".into()),
        2 => {
            // unary
            Expr::UnaryOp {
                op: crate::expr::UnaryOp::Neg,
                input: Box::new(random_expr(rng, max_depth - 1)),
            }
        }
        _ => {
            // binary
            let left = random_expr(rng, max_depth - 1);
            let right = random_expr(rng, max_depth - 1);
            let op = match rng.gen_range(0..4) {
                0 => crate::expr::BinaryOp::Add,
                1 => crate::expr::BinaryOp::Sub,
                2 => crate::expr::BinaryOp::Mul,
                _ => crate::expr::BinaryOp::Div,
            };
            Expr::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
    }
}

/// Run GA operating directly on Expr trees.
pub fn run_ga_exprs(
    config: &GAConfig,
    mut population: Vec<Expr>,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut impl Rng,
) -> (Expr, f64) {
    let pop_size = population.len();
    assert!(pop_size > 0);
    // initial evaluation (filter invalid)
    for e in population.iter_mut() {
        if !is_executable_expr(e) {
            *e = random_expr(rng, 2);
        }
    }
    let mut scores: Vec<f64> = population
        .par_iter()
        .map(|e| {
            if !is_executable_expr(e) {
                -1e9
            } else {
                evaluator.fitness(e)
            }
        })
        .collect();

    for _ in 0..config.generations {
        let mut new_pop: Vec<Expr> = Vec::with_capacity(pop_size);
        while new_pop.len() < pop_size {
            // tournament selection indices
            let mut best = rng.gen_range(0usize..pop_size);
            for _ in 1..config.tournament_size {
                let idx = rng.gen_range(0usize..pop_size);
                if scores[idx] > scores[best] {
                    best = idx;
                }
            }
            let mut best2 = rng.gen_range(0usize..pop_size);
            for _ in 1..config.tournament_size {
                let idx = rng.gen_range(0usize..pop_size);
                if scores[idx] > scores[best2] {
                    best2 = idx;
                }
            }
            let p1 = population[best].clone();
            let p2 = population[best2].clone();
            if rng.gen_bool(config.crossover_rate) {
                let (c1, c2) = subtree_crossover(&p1, &p2, rng);
                new_pop.push(c1);
                if new_pop.len() < pop_size {
                    new_pop.push(c2);
                }
            } else {
                new_pop.push(p1);
                if new_pop.len() < pop_size {
                    new_pop.push(p2);
                }
            }
        }
        // mutation
        for e in new_pop.iter_mut() {
            if rng.gen_bool(config.mutation_rate) {
                *e = subtree_mutate(e, 2, rng);
            }
        }
        // replace invalid
        for e in new_pop.iter_mut() {
            if !is_executable_expr(e) {
                *e = random_expr(rng, 2);
            }
        }
        scores = new_pop
            .par_iter()
            .map(|e| {
                if !is_executable_expr(e) {
                    -1e9
                } else {
                    evaluator.fitness(e)
                }
            })
            .collect();
        population = new_pop;
    }

    let mut best_idx = 0usize;
    for i in 1..population.len() {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }
    (population[best_idx].clone(), scores[best_idx])
}

/// Pipeline evaluator: evaluates Expr by building a Plan (single Project), running optimizer and executor
pub struct ShardedLruCache {
    shards: Vec<Mutex<LruCache<String, f64>>>,
    cap_per_shard: NonZeroUsize,
}

impl ShardedLruCache {
    pub fn with_shards(shards: usize, cap_per_shard: NonZeroUsize) -> Self {
        let mut v = Vec::with_capacity(shards);
        for _ in 0..shards {
            v.push(Mutex::new(LruCache::new(cap_per_shard)));
        }
        ShardedLruCache {
            shards: v,
            cap_per_shard,
        }
    }

    fn shard_for_key(&self, key: &str) -> usize {
        // simple hash to shard index using ahash
        let mut h = AHasher::default();
        h.write(key.as_bytes());
        (h.finish() as usize) % self.shards.len()
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        let i = self.shard_for_key(key);
        let mut s = self.shards[i].lock().unwrap();
        s.get(key).copied()
    }

    pub fn put(&self, key: String, val: f64) {
        let i = self.shard_for_key(&key);
        let mut s = self.shards[i].lock().unwrap();
        s.put(key, val);
    }

    pub fn iter_all(&self) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        for s in &self.shards {
            let guard = s.lock().unwrap();
            for (k, v) in guard.iter() {
                out.push((k.clone(), *v));
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.iter_all().len()
    }

    pub fn cap(&self) -> NonZeroUsize {
        self.cap_per_shard
    }

    pub fn shards_count(&self) -> usize {
        self.shards.len()
    }
}

pub struct PipelineEvaluator {
    pub optimizer: Arc<Optimizer>,
    pub provider: Arc<dyn DataProvider>,
    // cache: sharded LRU
    pub cache: ShardedLruCache,
    // Optional vector of execution contexts passed through to the executor (e.g., multiple (symbol,trading_date) maps)
    pub exec_ctx: Option<Vec<HashMap<String, String>>>,
}

impl PipelineEvaluator {
    /// Create with a default cache capacity
    pub fn new(opt: Arc<Optimizer>, provider: Arc<dyn DataProvider>) -> Self {
        Self::with_capacity(opt, provider, 64)
    }

    /// Create with explicit cache capacity
    pub fn with_capacity(
        opt: Arc<Optimizer>,
        provider: Arc<dyn DataProvider>,
        cache_capacity: usize,
    ) -> Self {
        // choose number of shards (use 1 shard to keep cache behavior deterministic in tests)
        let shards = 1usize;
        let per = (cache_capacity + shards - 1) / shards;
        let nz = NonZeroUsize::new(per.max(1)).unwrap();
        PipelineEvaluator {
            optimizer: opt,
            provider,
            cache: ShardedLruCache::with_shards(shards, nz),
            exec_ctx: None,
        }
    }

    /// Create from a `Config` (toml) structure. Uses config.cache_capacity and cache_shards when provided.
    pub fn with_config(
        opt: Arc<Optimizer>,
        provider: Arc<dyn DataProvider>,
        cfg: &crate::config::Config,
    ) -> Self {
        let capacity = cfg.cache_capacity.unwrap_or(64);
        let shards = cfg.cache_shards.unwrap_or(1usize);
        let per = (capacity + shards - 1) / shards;
        let nz = NonZeroUsize::new(per.max(1)).unwrap();
        PipelineEvaluator {
            optimizer: opt,
            provider,
            cache: ShardedLruCache::with_shards(shards, nz),
            exec_ctx: None,
        }
    }

    /// Construct a PipelineEvaluator from a Config. If ClickHouse config is present, this
    /// will create a ClickhouseProvider, enumerate group keys from `table` and set them as
    /// exec contexts. Returns Err if ClickHouse config exists but cannot be used.
    pub fn from_config(
        opt: Arc<Optimizer>,
        cfg: &crate::config::Config,
        table: &str,
    ) -> Result<Self, String> {
        // determine cache/shards
        let capacity = cfg.cache_capacity.unwrap_or(64);
        let shards = cfg.cache_shards.unwrap_or(1usize);
        let per = (capacity + shards - 1) / shards;
        let nz = NonZeroUsize::new(per.max(1)).unwrap();

        // attempt to build clickhouse provider from config
        if let Some(_) = &cfg.clickhouse {
            // build a concrete ClickhouseProvider, enumerate keys, then wrap in Arc<dyn DataProvider>
            let ch = crate::data_provider::ClickhouseProvider::from_config(cfg)?;
            let keys = ch.enumerate_group_keys(table)?;
            let mut contexts: Vec<HashMap<String, String>> = Vec::new();
            for (sym, td) in keys {
                let mut m = HashMap::new();
                m.insert("symbol".to_string(), sym);
                m.insert("trading_date".to_string(), td);
                contexts.push(m);
            }
            let provider_arc: Arc<dyn DataProvider> = Arc::new(ch);
            Ok(PipelineEvaluator {
                optimizer: opt,
                provider: provider_arc,
                cache: ShardedLruCache::with_shards(shards, nz),
                exec_ctx: Some(contexts),
            })
        } else {
            Err("clickhouse config not present in Config".to_string())
        }
    }

    /// Return a new evaluator with multiple execution contexts (each a map of keys like symbol/trading_date).
    pub fn with_exec_contexts(mut self, ctxs: Vec<HashMap<String, String>>) -> Self {
        self.exec_ctx = Some(ctxs);
        self
    }

    /// Set execution contexts on existing evaluator (mutable).
    pub fn set_exec_contexts(&mut self, ctxs: Vec<HashMap<String, String>>) {
        self.exec_ctx = Some(ctxs);
    }

    /// Compute a stable canonical string representation for Expr and hash it to a hex key.
    /// This avoids bincode allocation by traversing the AST deterministically.
    pub fn stable_key_for_expr(e: &Expr) -> String {
        // Allocation-minimizing canonical hashing:
        // write a normalized representation directly into the hasher.
        let mut hasher = AHasher::default();
        Self::write_normalized_to_hasher(&mut hasher, e);
        // produce a short hex key (u64)
        format!("{:016x}", hasher.finish())
    }

    /// Semantic normalization: constant folding and commutative associative normalization for Add/Mul
    pub fn semantic_normalize(e: &Expr) -> Expr {
        fn norm(e: &Expr) -> Expr {
            match e {
                Expr::Lit(_) | Expr::Col(_) => e.clone(),
                Expr::UnaryOp { op, input } => {
                    let ni = norm(input);
                    if let Expr::Lit(v) = ni {
                        match op {
                            crate::expr::UnaryOp::Neg => Expr::Lit(-v),
                        }
                    } else {
                        Expr::UnaryOp {
                            op: op.clone(),
                            input: Box::new(ni),
                        }
                    }
                }
                Expr::BinaryOp { op, left, right } => {
                    let nl = norm(left);
                    let nr = norm(right);
                    // constant folding
                    if let (Expr::Lit(lv), Expr::Lit(rv)) = (&nl, &nr) {
                        match op {
                            crate::expr::BinaryOp::Add => return Expr::Lit(lv + rv),
                            crate::expr::BinaryOp::Sub => return Expr::Lit(lv - rv),
                            crate::expr::BinaryOp::Mul => return Expr::Lit(lv * rv),
                            crate::expr::BinaryOp::Div => {
                                return if *rv != 0.0 {
                                    Expr::Lit(lv / rv)
                                } else {
                                    Expr::BinaryOp {
                                        op: op.clone(),
                                        left: Box::new(nl),
                                        right: Box::new(nr),
                                    }
                                };
                            }
                        }
                    }
                    // associative & commutative normalization for Add and Mul
                    match op {
                        crate::expr::BinaryOp::Add | crate::expr::BinaryOp::Mul => {
                            // gather operands by flattening nested same-op nodes
                            let mut ops: Vec<Expr> = Vec::new();
                            fn collect(
                                op_kind: &crate::expr::BinaryOp,
                                e: &Expr,
                                out: &mut Vec<Expr>,
                            ) {
                                match e {
                                    Expr::BinaryOp { op, left, right } if op == op_kind => {
                                        collect(op_kind, left, out);
                                        collect(op_kind, right, out);
                                    }
                                    other => out.push(other.clone()),
                                }
                            }
                            collect(op, &nl, &mut ops);
                            collect(op, &nr, &mut ops);
                            // fold constants among operands
                            let mut const_acc: Option<f64> = None;
                            let mut non_consts: Vec<Expr> = Vec::new();
                            for o in ops {
                                if let Expr::Lit(v) = o {
                                    const_acc = Some(match const_acc {
                                        Some(a) => {
                                            if matches!(op, crate::expr::BinaryOp::Add) {
                                                a + v
                                            } else {
                                                a * v
                                            }
                                        }
                                        None => v,
                                    });
                                } else {
                                    non_consts.push(o);
                                }
                            }
                            // if additive neutral 0 or multiplicative neutral 1 handle
                            if matches!(op, crate::expr::BinaryOp::Add) {
                                if let Some(c) = const_acc {
                                    if c != 0.0 {
                                        non_consts.push(Expr::Lit(c));
                                    }
                                }
                            } else {
                                if let Some(c) = const_acc {
                                    if c != 1.0 {
                                        non_consts.push(Expr::Lit(c));
                                    }
                                }
                            }
                            // sort non-const operands by debug string for stability
                            non_consts.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
                            // rebuild tree (left-associative)
                            if non_consts.is_empty() {
                                if let Some(c) = const_acc {
                                    return Expr::Lit(c);
                                } else {
                                    return Expr::Lit(0.0);
                                }
                            }
                            let mut it = non_consts.into_iter();
                            let mut acc = it.next().unwrap();
                            for n in it {
                                acc = Expr::BinaryOp {
                                    op: op.clone(),
                                    left: Box::new(acc),
                                    right: Box::new(n),
                                };
                            }
                            acc
                        }
                        _ => Expr::BinaryOp {
                            op: op.clone(),
                            left: Box::new(nl),
                            right: Box::new(nr),
                        },
                    }
                }
                Expr::Func { name, args } => {
                    let new_args: Vec<Expr> = args.iter().map(|a| norm(a)).collect();
                    Expr::Func {
                        name: name.clone(),
                        args: new_args,
                    }
                }
                Expr::Agg {
                    func,
                    input,
                    window,
                } => {
                    let ni = norm(input);
                    Expr::Agg {
                        func: func.clone(),
                        input: Box::new(ni),
                        window: window.clone(),
                    }
                }
            }
        }
        norm(e)
    }

    /// Write a canonical, normalized representation of `e` into the provided Sha256 hasher.
    /// This minimizes large temporary strings; it still computes per-operand digest bytes
    /// to provide a stable ordering for commutative ops.
    pub fn write_normalized_to_hasher<H: Hasher>(hasher: &mut H, e: &Expr) {
        match e {
            Expr::Col(name) => {
                hasher.write(b"Col:");
                hasher.write(name.as_bytes());
            }
            Expr::Lit(v) => {
                hasher.write(b"Lit:");
                hasher.write(&v.to_le_bytes());
            }
            Expr::UnaryOp { op, input } => {
                hasher.write(b"Unary:");
                match op {
                    crate::expr::UnaryOp::Neg => hasher.write(b"Neg:"),
                };
                Self::write_normalized_to_hasher(hasher, input);
            }
            Expr::BinaryOp { op, left, right } => {
                match op {
                    crate::expr::BinaryOp::Add | crate::expr::BinaryOp::Mul => {
                        // flatten same-op operands
                        let mut ops: Vec<Expr> = Vec::new();
                        fn collect(op_kind: &crate::expr::BinaryOp, e: &Expr, out: &mut Vec<Expr>) {
                            match e {
                                Expr::BinaryOp { op, left, right } if op == op_kind => {
                                    collect(op_kind, left, out);
                                    collect(op_kind, right, out);
                                }
                                other => out.push(other.clone()),
                            }
                        }
                        collect(op, left, &mut ops);
                        collect(op, right, &mut ops);
                        // compute digest bytes for each operand for stable ordering
                        let mut keyed: Vec<(Vec<u8>, Expr)> = Vec::new();
                        for o in ops.into_iter() {
                            let mut h = AHasher::default();
                            Self::write_normalized_to_hasher(&mut h, &o);
                            let keyb = h.finish().to_le_bytes().to_vec();
                            keyed.push((keyb, o));
                        }
                        keyed.sort_by(|a, b| a.0.cmp(&b.0));
                        // write op kind and ordered operands
                        hasher.write(b"Binary:");
                        match op {
                            crate::expr::BinaryOp::Add => hasher.write(b"Add:"),
                            crate::expr::BinaryOp::Mul => hasher.write(b"Mul:"),
                            _ => {}
                        }
                        for (kbytes, _) in keyed {
                            hasher.write(&kbytes);
                            hasher.write(&[b':']);
                        }
                    }
                    _ => {
                        hasher.write(b"Binary:");
                        match op {
                            crate::expr::BinaryOp::Sub => hasher.write(b"Sub:"),
                            crate::expr::BinaryOp::Div => hasher.write(b"Div:"),
                            _ => {}
                        }
                        Self::write_normalized_to_hasher(hasher, left);
                        hasher.write(&[b':']);
                        Self::write_normalized_to_hasher(hasher, right);
                    }
                }
            }
            Expr::Func { name, args } => {
                hasher.write(b"Func:");
                hasher.write(name.as_bytes());
                hasher.write(b"[");
                for a in args {
                    Self::write_normalized_to_hasher(hasher, a);
                    hasher.write(&[b',']);
                }
                hasher.write(b"]");
            }
            Expr::Agg {
                func,
                input,
                window,
            } => {
                hasher.write(b"Agg:");
                match func {
                    crate::expr::AggFunc::RollingMean => hasher.write(b"RollingMean:"),
                };
                Self::write_normalized_to_hasher(hasher, input);
                if let Some(w) = window {
                    hasher.write(b":W:");
                    hasher.write(&w.size.to_le_bytes());
                }
            }
        }
    }

    /// Versioned cache persistence format (see top-level `CacheDump`)

    /// Save cache contents to a file using bincode + version header
    pub fn save_cache_to_path(&self, path: &str) -> std::io::Result<()> {
        let items: Vec<(String, f64)> = self.cache.iter_all();
        let dump = CacheDump { version: 1, items };
        let bytes = bincode::serialize(&dump)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load cache contents from a file, replacing current cache (up to capacity). Older/newer versions should be handled in future.
    pub fn load_cache_from_path(&mut self, path: &str) -> std::io::Result<()> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let dump: CacheDump = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        if dump.version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported cache version",
            ));
        }
        let shards = self.cache.shards_count();
        let per = self.cache.cap().get();
        let new_sharded = ShardedLruCache::with_shards(shards, NonZeroUsize::new(per).unwrap());
        for (k, v) in dump.items.into_iter() {
            new_sharded.put(k, v);
        }
        self.cache = new_sharded;
        Ok(())
    }
}

impl FitnessEvaluator for PipelineEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        // Compute stable key based on semantic normalization and canonical hashing
        let key = Self::stable_key_for_expr(expr);
        {
            if let Some(v) = self.cache.get(&key) {
                return v;
            }
        }
        let plan = LogicalPlan::Projection {
            input: Arc::new(LogicalPlan::Scan {
                source_name: "s".into(),
                projection: None,
                filters: vec![],
            }),
            expr: vec![expr.clone()],
            schema: vec![("out".into(), crate::expr::DataType::Float)], // placeholder schema
        };
        let opt_plan = self.optimizer.optimize(plan);
        let exec = EvalExecutor;
        // If multiple execution contexts were provided, choose one deterministically based on the expr stable key.
        let res = match &self.exec_ctx {
            Some(ctxs) if !ctxs.is_empty() => {
                let mut h = AHasher::default();
                h.write(key.as_bytes());
                let idx = (h.finish() as usize) % ctxs.len();
                exec.run(&opt_plan, self.provider.as_ref(), Some(&ctxs[idx]))
            }
            _ => exec.run(&opt_plan, self.provider.as_ref(), None),
        };
        let val = res.value.unwrap_or(-1e9);
        self.cache.put(key, val);
        val
    }
}

#[derive(Debug, Clone)]
pub struct GenStat {
    pub avg_fitness: f64,
    pub replaced: usize,
}

#[derive(Debug, Clone)]
pub struct GAStats {
    pub gens: Vec<GenStat>,
}

/// Run a tree-based GA but also collect per-generation stats: average fitness and replacement count.
pub fn run_ga_exprs_with_stats(
    config: &GAConfig,
    mut population: Vec<Expr>,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut impl Rng,
) -> (Expr, f64, GAStats) {
    let pop_size = population.len();
    assert!(pop_size > 0);
    // initial replace invalid
    for e in population.iter_mut() {
        if !is_executable_expr(e) {
            *e = random_expr(rng, 2);
        }
    }
    let mut scores: Vec<f64> = population
        .par_iter()
        .map(|e| {
            if !is_executable_expr(e) {
                -1e9
            } else {
                evaluator.fitness(e)
            }
        })
        .collect();
    let mut stats = GAStats { gens: Vec::new() };

    for _gen in 0..config.generations {
        let mut replaced_count = 0usize;
        let mut new_pop: Vec<Expr> = Vec::with_capacity(pop_size);
        while new_pop.len() < pop_size {
            // tournament selection
            let mut best = rng.gen_range(0usize..pop_size);
            for _ in 1..config.tournament_size {
                let idx = rng.gen_range(0usize..pop_size);
                if scores[idx] > scores[best] {
                    best = idx;
                }
            }
            let mut best2 = rng.gen_range(0usize..pop_size);
            for _ in 1..config.tournament_size {
                let idx = rng.gen_range(0usize..pop_size);
                if scores[idx] > scores[best2] {
                    best2 = idx;
                }
            }
            let p1 = population[best].clone();
            let p2 = population[best2].clone();
            if rng.gen_bool(config.crossover_rate) {
                let (c1, c2) = subtree_crossover(&p1, &p2, rng);
                new_pop.push(c1);
                if new_pop.len() < pop_size {
                    new_pop.push(c2);
                }
            } else {
                new_pop.push(p1);
                if new_pop.len() < pop_size {
                    new_pop.push(p2);
                }
            }
        }
        // mutation
        for e in new_pop.iter_mut() {
            if rng.gen_bool(config.mutation_rate) {
                *e = subtree_mutate(e, 2, rng);
            }
        }
        // replace invalid and count
        for e in new_pop.iter_mut() {
            if !is_executable_expr(e) {
                *e = random_expr(rng, 2);
                replaced_count += 1;
            }
        }
        scores = new_pop
            .par_iter()
            .map(|e| {
                if !is_executable_expr(e) {
                    -1e9
                } else {
                    evaluator.fitness(e)
                }
            })
            .collect();
        let avg = scores.iter().copied().sum::<f64>() / scores.len() as f64;
        stats.gens.push(GenStat {
            avg_fitness: avg,
            replaced: replaced_count,
        });
        population = new_pop;
    }

    let mut best_idx = 0usize;
    for i in 1..population.len() {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }
    (population[best_idx].clone(), scores[best_idx], stats)
}
