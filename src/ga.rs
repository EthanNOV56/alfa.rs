//! Genetic algorithm framework for expression optimization
//!
//! This module provides generic genetic algorithm operations that can be used
//! for symbolic regression and other expression optimization tasks.

use crate::config::Config;
use crate::data_provider::DataProvider;
use crate::executor::{EvalExecutor, Executor};
use crate::expr::{Expr, Literal, BinaryOp, UnaryOp, AggregateOp, DataType};
use crate::logical_plan::LogicalPlan;
use crate::optimizer::{Optimizer, ConstantFolding, PredicatePushdown, ProjectionPushdown};
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
pub fn crossover<R: Rng + ?Sized>(a: &Chromosome, b: &Chromosome, rng: &mut R) -> (Chromosome, Chromosome) {
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
pub fn mutate<R: Rng + ?Sized>(ch: &mut Chromosome, rate: f64, max_val: u8, rng: &mut R) {
    for g in ch.genes.iter_mut() {
        if rng.gen_bool(rate) {
            *g = rng.gen_range(0..max_val);
        }
    }
}

fn random_chromosome<R: Rng + ?Sized>(len: usize, max_val: u8, rng: &mut R) -> Chromosome {
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
pub fn run_ga<R: Rng + ?Sized>(
    config: &GAConfig,
    mut population: Vec<Chromosome>,
    decoder: &dyn GenotypeDecoder,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut R,
) -> (Chromosome, f64) {
    let pop_size = population.len();
    let mut scores = evaluate_population(decoder, evaluator, &population);
    let mut best_idx = 0;
    for i in 1..pop_size {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }

    for _generation in 0..config.generations {
        // create next generation via tournament selection + crossover + mutation
        let mut next = Vec::with_capacity(pop_size);
        while next.len() < pop_size {
            // tournament selection
            let mut best = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let cand = rng.gen_range(0..pop_size);
                if scores[cand] > scores[best] {
                    best = cand;
                }
            }
            next.push(population[best].clone());
        }

        // crossover
        for i in (0..pop_size).step_by(2) {
            if i + 1 < pop_size && rng.gen_bool(config.crossover_rate) {
                let (c1, c2) = crossover(&next[i], &next[i + 1], rng);
                next[i] = c1;
                next[i + 1] = c2;
            }
        }

        // mutation
        for ch in next.iter_mut() {
            mutate(ch, config.mutation_rate, config.gene_max, rng);
        }

        population = next;
        scores = evaluate_population(decoder, evaluator, &population);
        for i in 0..pop_size {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }
    }
    (population[best_idx].clone(), scores[best_idx])
}

/// Basic check whether Expr contains unsupported nodes (e.g., unsupported function calls)
fn is_executable_expr(e: &Expr) -> bool {
    match e {
        // In alpha-expr, all function calls are supported via evaluation context
        // We'll assume all are executable
        Expr::FunctionCall { .. } => true,
        Expr::UnaryExpr { expr, .. } => is_executable_expr(expr),
        Expr::BinaryExpr { left, right, .. } => is_executable_expr(left) && is_executable_expr(right),
        Expr::Aggregate { expr, .. } => is_executable_expr(expr),
        Expr::Conditional { condition, then_expr, else_expr } => {
            is_executable_expr(condition) && is_executable_expr(then_expr) && is_executable_expr(else_expr)
        }
        Expr::Cast { expr, .. } => is_executable_expr(expr),
        _ => true,
    }
}

// --- Tree-based GA utilities ---

/// Collect all paths to nodes in the tree. A path is a sequence of child indices as described:
/// - Unary: child 0
/// - Binary: left 0, right 1
/// - FunctionCall: args indexed 0..n-1
/// - Conditional: condition 0, then 1, else 2
/// - Aggregate: expr 0
/// - Cast: expr 0
fn collect_paths(e: &Expr, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    out.push(cur.clone());
    match e {
        Expr::UnaryExpr { expr, .. } => {
            cur.push(0);
            collect_paths(expr, cur, out);
            cur.pop();
        }
        Expr::BinaryExpr { left, right, .. } => {
            cur.push(0);
            collect_paths(left, cur, out);
            cur.pop();
            
            cur.push(1);
            collect_paths(right, cur, out);
            cur.pop();
        }
        Expr::FunctionCall { args, .. } => {
            for (i, arg) in args.iter().enumerate() {
                cur.push(i);
                collect_paths(arg, cur, out);
                cur.pop();
            }
        }
        Expr::Aggregate { expr, .. } => {
            cur.push(0);
            collect_paths(expr, cur, out);
            cur.pop();
        }
        Expr::Conditional { condition, then_expr, else_expr } => {
            cur.push(0);
            collect_paths(condition, cur, out);
            cur.pop();
            
            cur.push(1);
            collect_paths(then_expr, cur, out);
            cur.pop();
            
            cur.push(2);
            collect_paths(else_expr, cur, out);
            cur.pop();
        }
        Expr::Cast { expr, .. } => {
            cur.push(0);
            collect_paths(expr, cur, out);
            cur.pop();
        }
        _ => {}
    }
}

/// Get node at path; returns None if path doesn't match tree shape
fn get_node_at_path<'a>(e: &'a Expr, path: &[usize]) -> Option<&'a Expr> {
    let mut cur = e;
    for &idx in path {
        cur = match cur {
            Expr::UnaryExpr { expr, .. } if idx == 0 => expr,
            Expr::BinaryExpr { left, right, .. } => {
                if idx == 0 {
                    left
                } else if idx == 1 {
                    right
                } else {
                    return None;
                }
            }
            Expr::FunctionCall { args, .. } => args.get(idx)?,
            Expr::Aggregate { expr, .. } if idx == 0 => expr,
            Expr::Conditional { condition, then_expr, else_expr, .. } => {
                if idx == 0 {
                    condition
                } else if idx == 1 {
                    then_expr
                } else if idx == 2 {
                    else_expr
                } else {
                    return None;
                }
            }
            Expr::Cast { expr, .. } if idx == 0 => expr,
            _ => return None,
        };
    }
    Some(cur)
}

/// Replace node at path with new subtree; returns new tree or None if path invalid
fn replace_node_at_path(e: Expr, path: &[usize], new: Expr) -> Option<Expr> {
    if path.is_empty() {
        return Some(new);
    }
    let idx = path[0];
    let rest = &path[1..];
    match e {
        Expr::UnaryExpr { op, expr } if idx == 0 => {
            let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
            Some(Expr::UnaryExpr { op, expr: Arc::new(new_expr) })
        }
        Expr::BinaryExpr { left, op, right } => {
            if idx == 0 {
                let new_left = replace_node_at_path((*left).clone(), rest, new)?;
                Some(Expr::BinaryExpr { left: Arc::new(new_left), op, right })
            } else if idx == 1 {
                let new_right = replace_node_at_path((*right).clone(), rest, new)?;
                Some(Expr::BinaryExpr { left, op, right: Arc::new(new_right) })
            } else {
                None
            }
        }
        Expr::FunctionCall { name, args } => {
            let mut new_args = args;
            if idx < new_args.len() {
                let new_arg = replace_node_at_path(new_args[idx].clone(), rest, new)?;
                new_args[idx] = new_arg;
                Some(Expr::FunctionCall { name, args: new_args })
            } else {
                None
            }
        }
        Expr::Aggregate { op, expr, distinct } if idx == 0 => {
            let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
            Some(Expr::Aggregate { op, expr: Arc::new(new_expr), distinct })
        }
        Expr::Conditional { condition, then_expr, else_expr } => {
            let new_expr = if idx == 0 {
                let new_cond = replace_node_at_path((*condition).clone(), rest, new)?;
                Expr::Conditional { 
                    condition: Arc::new(new_cond), 
                    then_expr, 
                    else_expr 
                }
            } else if idx == 1 {
                let new_then = replace_node_at_path((*then_expr).clone(), rest, new)?;
                Expr::Conditional { 
                    condition, 
                    then_expr: Arc::new(new_then), 
                    else_expr 
                }
            } else if idx == 2 {
                let new_else = replace_node_at_path((*else_expr).clone(), rest, new)?;
                Expr::Conditional { 
                    condition, 
                    then_expr, 
                    else_expr: Arc::new(new_else) 
                }
            } else {
                return None;
            };
            Some(new_expr)
        }
        Expr::Cast { expr, data_type } if idx == 0 => {
            let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
            Some(Expr::Cast { expr: Arc::new(new_expr), data_type })
        }
        _ => None,
    }
}

/// Subtree crossover for expression trees
pub fn subtree_crossover<R: Rng + ?Sized>(a: &Expr, b: &Expr, rng: &mut R) -> (Expr, Expr) {
    // collect all paths in both trees
    let mut paths_a = Vec::new();
    collect_paths(a, &mut Vec::new(), &mut paths_a);
    let mut paths_b = Vec::new();
    collect_paths(b, &mut Vec::new(), &mut paths_b);

    // filter out root path (empty) and leaf-only paths if desired
    paths_a.retain(|p| !p.is_empty());
    paths_b.retain(|p| !p.is_empty());

    if paths_a.is_empty() || paths_b.is_empty() {
        return (a.clone(), b.clone());
    }

    // pick random path in each
    let pa = paths_a[rng.gen_range(0..paths_a.len())].clone();
    let pb = paths_b[rng.gen_range(0..paths_b.len())].clone();

    // extract subtrees
    let sa = get_node_at_path(a, &pa);
    let sb = get_node_at_path(b, &pb);

    match (sa, sb) {
        (Some(sub_a), Some(sub_b)) => {
            // replace subtree in a with subtree from b, and vice versa
            let new_a = replace_node_at_path(a.clone(), &pa, sub_b.clone());
            let new_b = replace_node_at_path(b.clone(), &pb, sub_a.clone());
            (new_a.unwrap_or_else(|| a.clone()), new_b.unwrap_or_else(|| b.clone()))
        }
        _ => (a.clone(), b.clone()),
    }
}

/// Subtree mutation: replace a random subtree with a newly generated one
pub fn subtree_mutate<R: Rng + ?Sized>(e: &Expr, _max_depth: usize, rng: &mut R) -> Expr {
    let mut paths = Vec::new();
    collect_paths(e, &mut Vec::new(), &mut paths);
    paths.retain(|p| !p.is_empty());
    if paths.is_empty() {
        return e.clone();
    }
    let p = paths[rng.gen_range(0..paths.len())].clone();
    
    // For simplicity, generate a random literal as replacement
    // In a real implementation, you'd have a function to generate random expressions
    let new_subtree = Expr::Literal(Literal::Float(rng.gen_range(0.0..100.0)));
    
    replace_node_at_path(e.clone(), &p, new_subtree).unwrap_or_else(|| e.clone())
}

/// Run GA directly on expression trees (no chromosome encoding)
pub fn run_ga_exprs<R: Rng + ?Sized>(
    config: &GAConfig,
    initial: Vec<Expr>,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut R,
) -> (Expr, f64) {
    let pop_size = initial.len();
    let mut population = initial;
    let mut scores: Vec<f64> = population
        .iter()
        .map(|e| {
            if !is_executable_expr(e) {
                -1e9
            } else {
                evaluator.fitness(e)
            }
        })
        .collect();
    let mut best_idx = 0;
    for i in 1..pop_size {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }

    for _gen in 0..config.generations {
        let mut next = Vec::with_capacity(pop_size);
        while next.len() < pop_size {
            // tournament selection
            let mut best = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let cand = rng.gen_range(0..pop_size);
                if scores[cand] > scores[best] {
                    best = cand;
                }
            }
            next.push(population[best].clone());
        }

        // crossover
        for i in (0..pop_size).step_by(2) {
            if i + 1 < pop_size && rng.gen_bool(config.crossover_rate) {
                let (c1, c2) = subtree_crossover(&next[i], &next[i + 1], rng);
                next[i] = c1;
                next[i + 1] = c2;
            }
        }

        // mutation (with a low probability per individual)
        for expr in next.iter_mut() {
            if rng.gen_bool(config.mutation_rate) {
                *expr = subtree_mutate(expr, 5, rng);
            }
        }

        population = next;
        scores = population
            .iter()
            .map(|e| {
                if !is_executable_expr(e) {
                    -1e9
                } else {
                    evaluator.fitness(e)
                }
            })
            .collect();
        for i in 0..pop_size {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }
    }
    (population[best_idx].clone(), scores[best_idx])
}

// --- Caching infrastructure ---

/// Sharded LRU cache for concurrent fitness evaluation
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

/// Pipeline evaluator: evaluates Expr by building a Plan (single Project), running optimizer and executor
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
        cfg: &Config,
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
        cfg: &Config,
        table: &str,
    ) -> Result<Self, String> {
        // determine cache/shards
        let capacity = cfg.cache_capacity.unwrap_or(64);
        let shards = cfg.cache_shards.unwrap_or(1usize);
        let per = (capacity + shards - 1) / shards;
        let nz = NonZeroUsize::new(per.max(1)).unwrap();

        // attempt to build clickhouse provider from config
        if let Some(ch_cfg) = &cfg.clickhouse {
            // Use the clickhouse_provider module
            use crate::clickhouse_provider::ClickhouseProvider;
            let ch = ClickhouseProvider::from_config(ch_cfg)?;
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
                Expr::Literal(_) | Expr::Column(_) => e.clone(),
                Expr::UnaryExpr { op, expr } => {
                    let ni = norm(expr);
                    if let Expr::Literal(Literal::Float(v)) = &ni {
                        match op {
                            UnaryOp::Negate => Expr::Literal(Literal::Float(-v)),
                            _ => Expr::UnaryExpr { op: op.clone(), expr: Arc::new(ni) },
                        }
                    } else {
                        Expr::UnaryExpr { op: op.clone(), expr: Arc::new(ni) }
                    }
                }
                Expr::BinaryExpr { op, left, right } => {
                    let nl = norm(left);
                    let nr = norm(right);
                    // constant folding
                    if let (Expr::Literal(Literal::Float(lv)), Expr::Literal(Literal::Float(rv))) = (&nl, &nr) {
                        match op {
                            BinaryOp::Add => return Expr::Literal(Literal::Float(lv + rv)),
                            BinaryOp::Subtract => return Expr::Literal(Literal::Float(lv - rv)),
                            BinaryOp::Multiply => return Expr::Literal(Literal::Float(lv * rv)),
                            BinaryOp::Divide => {
                                return if *rv != 0.0 {
                                    Expr::Literal(Literal::Float(lv / rv))
                                } else {
                                    Expr::BinaryExpr {
                                        op: op.clone(),
                                        left: Arc::new(nl),
                                        right: Arc::new(nr),
                                    }
                                };
                            }
                            _ => {}
                        }
                    }
                    // associative & commutative normalization for Add and Mul
                    match op {
                        BinaryOp::Add | BinaryOp::Multiply => {
                            // gather operands by flattening nested same-op nodes
                            let mut ops: Vec<Expr> = Vec::new();
                            fn collect(
                                op_kind: &BinaryOp,
                                e: &Expr,
                                out: &mut Vec<Expr>,
                            ) {
                                match e {
                                    Expr::BinaryExpr { op, left, right } if op == op_kind => {
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
                                if let Expr::Literal(Literal::Float(v)) = o {
                                    const_acc = Some(match const_acc {
                                        Some(a) => {
                                            if matches!(op, BinaryOp::Add) {
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
                            if matches!(op, BinaryOp::Add) {
                                if let Some(c) = const_acc {
                                    if c != 0.0 {
                                        non_consts.push(Expr::Literal(Literal::Float(c)));
                                    }
                                }
                            } else {
                                if let Some(c) = const_acc {
                                    if c != 1.0 {
                                        non_consts.push(Expr::Literal(Literal::Float(c)));
                                    }
                                }
                            }
                            // sort non-const operands by debug string for stability
                            non_consts.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
                            // rebuild tree (left-associative)
                            if non_consts.is_empty() {
                                if let Some(c) = const_acc {
                                    return Expr::Literal(Literal::Float(c));
                                } else {
                                    return Expr::Literal(Literal::Float(0.0));
                                }
                            }
                            let mut it = non_consts.into_iter();
                            let mut acc = it.next().unwrap();
                            for n in it {
                                acc = Expr::BinaryExpr {
                                    op: op.clone(),
                                    left: Arc::new(acc),
                                    right: Arc::new(n),
                                };
                            }
                            acc
                        }
                        _ => Expr::BinaryExpr {
                            op: op.clone(),
                            left: Arc::new(nl),
                            right: Arc::new(nr),
                        },
                    }
                }
                Expr::FunctionCall { name, args } => {
                    let new_args: Vec<Expr> = args.iter().map(|a| norm(a)).collect();
                    Expr::FunctionCall {
                        name: name.clone(),
                        args: new_args,
                    }
                }
                Expr::Aggregate { op, expr, distinct } => {
                    let ni = norm(expr);
                    Expr::Aggregate {
                        op: op.clone(),
                        expr: Arc::new(ni),
                        distinct: *distinct,
                    }
                }
                Expr::Conditional { condition, then_expr, else_expr } => {
                    let nc = norm(condition);
                    let nt = norm(then_expr);
                    let ne = norm(else_expr);
                    Expr::Conditional {
                        condition: Arc::new(nc),
                        then_expr: Arc::new(nt),
                        else_expr: Arc::new(ne),
                    }
                }
                Expr::Cast { expr, data_type } => {
                    let ne = norm(expr);
                    Expr::Cast {
                        expr: Arc::new(ne),
                        data_type: data_type.clone(),
                    }
                }
            }
        }
        norm(e)
    }

    /// Write a canonical, normalized representation of `e` into the provided hasher.
    /// This minimizes large temporary strings; it still computes per-operand digest bytes
    /// to provide a stable ordering for commutative ops.
    pub fn write_normalized_to_hasher<H: Hasher>(hasher: &mut H, e: &Expr) {
        match e {
            Expr::Column(name) => {
                hasher.write(b"Col:");
                hasher.write(name.as_bytes());
            }
            Expr::Literal(Literal::Float(v)) => {
                hasher.write(b"Float:");
                // Normalize float representation: write as integer if possible
                if v.fract() == 0.0 {
                    hasher.write(format!("{}", *v as i64).as_bytes());
                } else {
                    hasher.write(format!("{:.6}", v).as_bytes());
                }
            }
            Expr::Literal(Literal::Integer(i)) => {
                hasher.write(b"Int:");
                hasher.write(format!("{}", i).as_bytes());
            }
            Expr::Literal(Literal::Boolean(b)) => {
                hasher.write(b"Bool:");
                hasher.write(if *b { b"true" } else { b"false" });
            }
            Expr::Literal(Literal::String(s)) => {
                hasher.write(b"Str:");
                hasher.write(s.as_bytes());
            }
            Expr::Literal(Literal::Null) => {
                hasher.write(b"Null");
            }
            Expr::UnaryExpr { op, expr } => {
                hasher.write(b"Unary:");
                match op {
                    UnaryOp::Negate => hasher.write(b"Negate"),
                    UnaryOp::Not => hasher.write(b"Not"),
                    UnaryOp::Abs => hasher.write(b"Abs"),
                    UnaryOp::Sqrt => hasher.write(b"Sqrt"),
                    UnaryOp::Log => hasher.write(b"Log"),
                    UnaryOp::Exp => hasher.write(b"Exp"),
                }
                hasher.write(b"(");
                Self::write_normalized_to_hasher(hasher, expr);
                hasher.write(b")");
            }
            Expr::BinaryExpr { op, left, right } => {
                hasher.write(b"Binary:");
                match op {
                    BinaryOp::Add => hasher.write(b"Add"),
                    BinaryOp::Subtract => hasher.write(b"Sub"),
                    BinaryOp::Multiply => hasher.write(b"Mul"),
                    BinaryOp::Divide => hasher.write(b"Div"),
                    BinaryOp::Modulo => hasher.write(b"Mod"),
                    BinaryOp::Equal => hasher.write(b"Eq"),
                    BinaryOp::NotEqual => hasher.write(b"Neq"),
                    BinaryOp::GreaterThan => hasher.write(b"Gt"),
                    BinaryOp::GreaterThanOrEqual => hasher.write(b"Ge"),
                    BinaryOp::LessThan => hasher.write(b"Lt"),
                    BinaryOp::LessThanOrEqual => hasher.write(b"Le"),
                    BinaryOp::And => hasher.write(b"And"),
                    BinaryOp::Or => hasher.write(b"Or"),
                }
                // For commutative ops (Add, Mul), sort operand hashes
                if matches!(op, BinaryOp::Add | BinaryOp::Multiply) {
                    let mut left_hash = AHasher::default();
                    Self::write_normalized_to_hasher(&mut left_hash, left);
                    let left_digest = left_hash.finish();
                    
                    let mut right_hash = AHasher::default();
                    Self::write_normalized_to_hasher(&mut right_hash, right);
                    let right_digest = right_hash.finish();
                    
                    if left_digest <= right_digest {
                        hasher.write_u64(left_digest);
                        hasher.write_u64(right_digest);
                    } else {
                        hasher.write_u64(right_digest);
                        hasher.write_u64(left_digest);
                    }
                } else {
                    Self::write_normalized_to_hasher(hasher, left);
                    Self::write_normalized_to_hasher(hasher, right);
                }
                hasher.write(b")");
            }
            Expr::FunctionCall { name, args } => {
                hasher.write(b"Func:");
                hasher.write(name.as_bytes());
                hasher.write(b"[");
                for arg in args {
                    Self::write_normalized_to_hasher(hasher, arg);
                    hasher.write(b",");
                }
                hasher.write(b"]");
            }
            Expr::Aggregate { op, expr, distinct } => {
                hasher.write(b"Agg:");
                match op {
                    AggregateOp::Sum => hasher.write(b"Sum"),
                    AggregateOp::Mean => hasher.write(b"Mean"),
                    AggregateOp::Min => hasher.write(b"Min"),
                    AggregateOp::Max => hasher.write(b"Max"),
                    AggregateOp::Count => hasher.write(b"Count"),
                    AggregateOp::StdDev => hasher.write(b"StdDev"),
                    AggregateOp::Variance => hasher.write(b"Variance"),
                }
                if *distinct {
                    hasher.write(b"D");
                }
                hasher.write(b"(");
                Self::write_normalized_to_hasher(hasher, expr);
                hasher.write(b")");
            }
            Expr::Conditional { condition, then_expr, else_expr } => {
                hasher.write(b"If:");
                Self::write_normalized_to_hasher(hasher, condition);
                hasher.write(b"?");
                Self::write_normalized_to_hasher(hasher, then_expr);
                hasher.write(b":");
                Self::write_normalized_to_hasher(hasher, else_expr);
            }
            Expr::Cast { expr, data_type } => {
                hasher.write(b"Cast:");
                Self::write_normalized_to_hasher(hasher, expr);
                hasher.write(b"->");
                match data_type {
                    DataType::Boolean => hasher.write(b"bool"),
                    DataType::Integer => hasher.write(b"int"),
                    DataType::Float => hasher.write(b"float"),
                    DataType::String => hasher.write(b"string"),
                }
            }
        }
    }

    /// Fitness evaluation with caching
    pub fn fitness(&self, expr: &Expr) -> f64 {
        let key = Self::stable_key_for_expr(expr);
        // Try cache first (with a short critical section)
        {
            if let Some(v) = self.cache.get(&key) {
                return v;
            }
        }
        // Build a simple logical plan with the expression
        let plan = LogicalPlan::Projection {
            input: Arc::new(LogicalPlan::Scan {
                source_name: "s".into(),
                projection: None,
                filters: vec![],
            }),
            expr: vec![expr.clone()],
            schema: vec![("out".into(), DataType::Float)],
        };
        let opt_plan = self.optimizer.optimize(plan);
        let exec = EvalExecutor;
        // If multiple execution contexts were provided, choose one deterministically based on the expr stable key.
        let res = match &self.exec_ctx {
            Some(ctxs) if !ctxs.is_empty() => {
                let mut h = AHasher::default();
                h.write(key.as_bytes());
                let idx = (h.finish() as usize) % ctxs.len();
                exec.execute_plan(&opt_plan, self.provider.as_ref(), Some(&ctxs[idx]))
            }
            _ => exec.execute_plan(&opt_plan, self.provider.as_ref(), None),
        };
        let val = res.value.unwrap_or(-1e9);
        self.cache.put(key, val);
        val
    }

    /// Save cache contents to a file (serialized with bincode)
    pub fn save_cache_to_path(&self, path: &str) -> std::io::Result<()> {
        let items = self.cache.iter_all();
        let dump = CacheDump {
            version: 1,
            items,
        };
        let encoded = bincode::serialize(&dump).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Load cache contents from a file
    pub fn load_cache_from_path(&mut self, path: &str) -> std::io::Result<()> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let dump: CacheDump = bincode::deserialize(&buf).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        for (key, val) in dump.items {
            self.cache.put(key, val);
        }
        Ok(())
    }
}

impl FitnessEvaluator for PipelineEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        self.fitness(expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    
    #[test]
    fn test_chromosome_crossover() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = Chromosome { genes: vec![1, 2, 3, 4] };
        let b = Chromosome { genes: vec![5, 6, 7, 8] };
        let (c1, c2) = crossover(&a, &b, &mut rng);
        assert_eq!(c1.genes.len(), 4);
        assert_eq!(c2.genes.len(), 4);
    }
    
    #[test]
    fn test_chromosome_mutate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ch = Chromosome { genes: vec![1, 2, 3, 4] };
        mutate(&mut ch, 0.5, 10, &mut rng);
        assert_eq!(ch.genes.len(), 4);
    }
    
    #[test]
    fn test_sharded_cache() {
        let cache = ShardedLruCache::with_shards(2, NonZeroUsize::new(10).unwrap());
        assert_eq!(cache.shards_count(), 2);
        
        cache.put("key1".to_string(), 42.0);
        assert_eq!(cache.get("key1"), Some(42.0));
        assert_eq!(cache.get("key2"), None);
        
        cache.put("key2".to_string(), 100.0);
        assert_eq!(cache.get("key2"), Some(100.0));
        
        let items = cache.iter_all();
        assert!(items.len() >= 2);
    }
    
    #[test]
    fn test_pipeline_evaluator_creation() {
        let optimizer = Arc::new(Optimizer::new()
            .add_rule(ConstantFolding::new())
            .add_rule(PredicatePushdown::new())
            .add_rule(ProjectionPushdown::new()));
        let provider = Arc::new(MockProvider::new(HashMap::new()));
        
        let evaluator = PipelineEvaluator::new(optimizer, provider);
        assert_eq!(evaluator.cache.shards_count(), 1);
    }
    
    #[test]
    fn test_stable_key_for_expr() {
        let expr = Expr::Literal(Literal::Float(10.0))
            .add(Expr::Literal(Literal::Float(20.0)));
        let key = PipelineEvaluator::stable_key_for_expr(&expr);
        assert!(!key.is_empty());
        
        // Same expression should produce same key
        let expr2 = Expr::Literal(Literal::Float(10.0))
            .add(Expr::Literal(Literal::Float(20.0)));
        let key2 = PipelineEvaluator::stable_key_for_expr(&expr2);
        assert_eq!(key, key2);
    }
    
    #[test]
    fn test_semantic_normalize() {
        // Test constant folding
        let expr = Expr::Literal(Literal::Float(10.0))
            .add(Expr::Literal(Literal::Float(20.0)));
        let norm = PipelineEvaluator::semantic_normalize(&expr);
        if let Expr::Literal(Literal::Float(v)) = norm {
            assert_eq!(v, 30.0);
        } else {
            panic!("Expected literal 30.0");
        }
        
        // Test commutative normalization
        let expr = Expr::Literal(Literal::Float(5.0))
            .add(Expr::Column("x".to_string()));
        let norm = PipelineEvaluator::semantic_normalize(&expr);
        // Should keep the same structure
        assert!(matches!(norm, Expr::BinaryExpr { .. }));
    }
}