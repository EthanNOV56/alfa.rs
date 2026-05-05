//! Tree operators, structural similarity analysis, and mutation templates.
//!
//! This module consolidates all expression-tree manipulation: subtree crossover,
//! mutation (including frequency-aware variants), smart template mutations,
//! and structural similarity / redundancy checks.

use crate::expr::Expr;
use rand::Rng;
use std::hash::{Hash, Hasher};

/// Tree operations for genetic programming
pub mod tree_ops {
    use crate::data::frequency;
    use crate::expr::Expr;
    use crate::expr::ast::Frequency;
    use crate::gp::generator::ExpressionGenerator;
    use crate::gp::types::OperatorFeedback;
    use rand::Rng;
    use std::sync::Arc;

    /// Collect all paths to nodes in the expression tree
    pub fn collect_paths(e: &Expr, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
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
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
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

    /// Get node at path
    pub fn get_node_at_path<'a>(e: &'a Expr, path: &[usize]) -> Option<&'a Expr> {
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
                Expr::Conditional {
                    condition,
                    then_expr,
                    else_expr,
                    ..
                } => {
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

    /// Replace node at path with new subtree
    pub fn replace_node_at_path(e: Expr, path: &[usize], new: Expr) -> Option<Expr> {
        if path.is_empty() {
            return Some(new);
        }

        let idx = path[0];
        let rest = &path[1..];

        match e {
            Expr::UnaryExpr { op, expr } if idx == 0 => {
                let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
                Some(Expr::UnaryExpr {
                    op,
                    expr: Arc::new(new_expr),
                })
            }
            Expr::BinaryExpr { left, op, right } => {
                if idx == 0 {
                    let new_left = replace_node_at_path((*left).clone(), rest, new)?;
                    Some(Expr::BinaryExpr {
                        left: Arc::new(new_left),
                        op,
                        right,
                    })
                } else if idx == 1 {
                    let new_right = replace_node_at_path((*right).clone(), rest, new)?;
                    Some(Expr::BinaryExpr {
                        left,
                        op,
                        right: Arc::new(new_right),
                    })
                } else {
                    None
                }
            }
            Expr::FunctionCall { name, args, freq } => {
                let mut new_args = args;
                if idx < new_args.len() {
                    let new_arg = replace_node_at_path(new_args[idx].clone(), rest, new)?;
                    new_args[idx] = new_arg;
                    Some(Expr::FunctionCall {
                        name,
                        args: new_args,
                        freq,
                    })
                } else {
                    None
                }
            }
            Expr::Aggregate { op, expr, distinct } if idx == 0 => {
                let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
                Some(Expr::Aggregate {
                    op,
                    expr: Arc::new(new_expr),
                    distinct,
                })
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let new_expr = if idx == 0 {
                    let new_cond = replace_node_at_path((*condition).clone(), rest, new)?;
                    Expr::Conditional {
                        condition: Arc::new(new_cond),
                        then_expr,
                        else_expr,
                    }
                } else if idx == 1 {
                    let new_then = replace_node_at_path((*then_expr).clone(), rest, new)?;
                    Expr::Conditional {
                        condition,
                        then_expr: Arc::new(new_then),
                        else_expr,
                    }
                } else if idx == 2 {
                    let new_else = replace_node_at_path((*else_expr).clone(), rest, new)?;
                    Expr::Conditional {
                        condition,
                        then_expr,
                        else_expr: Arc::new(new_else),
                    }
                } else {
                    return None;
                };
                Some(new_expr)
            }
            Expr::Cast { expr, data_type } if idx == 0 => {
                let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
                Some(Expr::Cast {
                    expr: Arc::new(new_expr),
                    data_type,
                })
            }
            _ => None,
        }
    }

    /// Subtree crossover for expression trees
    pub fn subtree_crossover<R: Rng + ?Sized>(a: &Expr, b: &Expr, rng: &mut R) -> (Expr, Expr) {
        // Collect all paths in both trees
        let mut paths_a = Vec::new();
        collect_paths(a, &mut Vec::new(), &mut paths_a);
        let mut paths_b = Vec::new();
        collect_paths(b, &mut Vec::new(), &mut paths_b);

        // Filter out root path (empty)
        paths_a.retain(|p| !p.is_empty());
        paths_b.retain(|p| !p.is_empty());

        if paths_a.is_empty() || paths_b.is_empty() {
            return (a.clone(), b.clone());
        }

        // Pick random path in each
        let pa = paths_a[rng.gen_range(0..paths_a.len())].clone();
        let pb = paths_b[rng.gen_range(0..paths_b.len())].clone();

        // Extract subtrees
        let sa = get_node_at_path(a, &pa);
        let sb = get_node_at_path(b, &pb);

        match (sa, sb) {
            (Some(sub_a), Some(sub_b)) => {
                // Replace subtree in a with subtree from b, and vice versa
                let new_a = replace_node_at_path(a.clone(), &pa, sub_b.clone());
                let new_b = replace_node_at_path(b.clone(), &pb, sub_a.clone());
                (
                    new_a.unwrap_or_else(|| a.clone()),
                    new_b.unwrap_or_else(|| b.clone()),
                )
            }
            _ => (a.clone(), b.clone()),
        }
    }

    /// Subtree mutation
    pub fn subtree_mutate<R: Rng + ?Sized>(
        e: &Expr,
        generator: &ExpressionGenerator<'_>,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        let mut paths = Vec::new();
        collect_paths(e, &mut Vec::new(), &mut paths);
        paths.retain(|p| !p.is_empty());

        if paths.is_empty() {
            return e.clone();
        }

        let p = paths[rng.gen_range(0..paths.len())].clone();
        let new_subtree = generator.generate_random_expr(max_depth, rng);

        replace_node_at_path(e.clone(), &p, new_subtree).unwrap_or_else(|| e.clone())
    }

    /// Subtree mutation with operator-feedback-biased subtree generation.
    pub(crate) fn subtree_mutate_feedback<R: Rng + ?Sized>(
        e: &Expr,
        generator: &ExpressionGenerator<'_>,
        feedback: &OperatorFeedback,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        let mut paths = Vec::new();
        collect_paths(e, &mut Vec::new(), &mut paths);
        paths.retain(|p| !p.is_empty());

        if paths.is_empty() {
            return e.clone();
        }

        let p = paths[rng.gen_range(0..paths.len())].clone();
        let new_subtree = generator.generate_feedback_expr(feedback, max_depth, rng);

        replace_node_at_path(e.clone(), &p, new_subtree).unwrap_or_else(|| e.clone())
    }

    /// Frequency mutation: change the freq annotation on a random FunctionCall.
    pub fn mutate_frequency<R: Rng + ?Sized>(e: &Expr, rng: &mut R) -> Expr {
        let mut paths = Vec::new();
        collect_paths(e, &mut Vec::new(), &mut paths);
        let funcs: Vec<&Vec<usize>> = paths
            .iter()
            .filter(|p| {
                !p.is_empty() && matches!(get_node_at_path(e, p), Some(Expr::FunctionCall { .. }))
            })
            .collect();
        if funcs.is_empty() {
            return e.clone();
        }
        let p = funcs[rng.gen_range(0..funcs.len())].clone();
        if let Some(Expr::FunctionCall {
            name,
            args,
            freq: old,
        }) = get_node_at_path(e, &p)
        {
            let mut candidates: Vec<Option<Frequency>> = Vec::new();
            if let Some(df) = args.first().and_then(|a| frequency::infer_frequency(a)) {
                for f in frequency::all_frequencies() {
                    if frequency::can_aggregate(f, &df) {
                        candidates.push(Some(f.clone()));
                    }
                }
            }
            candidates.push(None);
            if candidates.len() <= 1 {
                return e.clone();
            }
            let new_freq = candidates[rng.gen_range(0..candidates.len())].clone();
            if &new_freq != old {
                let node = Expr::FunctionCall {
                    name: name.clone(),
                    args: args.clone(),
                    freq: new_freq,
                };
                return replace_node_at_path(e.clone(), &p, node).unwrap_or_else(|| e.clone());
            }
        }
        e.clone()
    }
}

/// Compute a family hash for structural deduplication.
/// Two expressions that differ only in numeric constants produce the same hash.
pub(crate) fn family_hash(expr: &Expr) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let normalized = crate::gp::types::normalize_expression(expr);
    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

/// Apply a constrained smart mutation template.
/// A is the current expression (main signal), B is a random peer from the population (modifier).
/// Returns the mutated expression.
pub(crate) fn apply_smart_mutation<R: Rng + ?Sized>(a: &Expr, b: &Expr, rng: &mut R) -> Expr {
    match rng.gen_range(0..4) {
        // Weak gate: rank(B) > 0.55 ? A : -1
        0 => {
            let rank_b = Expr::function("rank", vec![b.clone()]);
            let cond = rank_b.gt(Expr::lit_float(0.55));
            Expr::conditional(cond, a.clone(), Expr::lit_float(-1.0))
        }
        // Regime conditional: if_else(ts_rank(B, 20) > 0.5, A, 0.5*A)
        1 => {
            let ts_rank_b = Expr::function("ts_rank", vec![b.clone(), Expr::lit_int(20)]);
            let cond = ts_rank_b.gt(Expr::lit_float(0.5));
            let damped = a.clone().mul(Expr::lit_float(0.5));
            Expr::conditional(cond, a.clone(), damped)
        }
        // Heterogeneous injection: A + 0.1 * rank(B)
        2 => {
            let rank_b = Expr::function("rank", vec![b.clone()]);
            let weak = rank_b.mul(Expr::lit_float(0.1));
            a.clone().add(weak)
        }
        // Cross-family combination: rank(A) * 0.9 + rank(B) * 0.1
        _ => {
            let rank_a = Expr::function("rank", vec![a.clone()]);
            let rank_b = Expr::function("rank", vec![b.clone()]);
            rank_a
                .mul(Expr::lit_float(0.9))
                .add(rank_b.mul(Expr::lit_float(0.1)))
        }
    }
}

/// Maximum Common Isomorphic Subtree size between two expression trees.
/// Counts the maximum number of structurally-equivalent nodes shared.
/// Constants of the same type (Float/Integer) are considered structurally equal.
fn mcis_size(a: &Expr, b: &Expr) -> usize {
    if same_structure_root(a, b) {
        match (a, b) {
            (Expr::Literal(_), Expr::Literal(_)) => 1,
            (Expr::Column(_), Expr::Column(_)) => 1,
            (Expr::UnaryExpr { expr: ae, .. }, Expr::UnaryExpr { expr: be, .. }) => {
                1 + mcis_size(ae, be)
            }
            (
                Expr::BinaryExpr {
                    left: al,
                    op: _,
                    right: ar,
                },
                Expr::BinaryExpr {
                    left: bl,
                    op: _,
                    right: br,
                },
            ) => 1 + mcis_size(al, bl) + mcis_size(ar, br),
            (
                Expr::FunctionCall {
                    name: an,
                    args: aa,
                    freq: _,
                },
                Expr::FunctionCall {
                    name: bn,
                    args: ba,
                    freq: _,
                },
            ) if an == bn && aa.len() == ba.len() => {
                1 + aa
                    .iter()
                    .zip(ba.iter())
                    .map(|(a, b)| mcis_size(a, b))
                    .sum::<usize>()
            }
            (
                Expr::Conditional {
                    condition: ac,
                    then_expr: at,
                    else_expr: ae,
                },
                Expr::Conditional {
                    condition: bc,
                    then_expr: bt,
                    else_expr: be,
                },
            ) => 1 + mcis_size(ac, bc) + mcis_size(at, bt) + mcis_size(ae, be),
            (Expr::Aggregate { expr: ae, .. }, Expr::Aggregate { expr: be, .. }) => {
                1 + mcis_size(ae, be)
            }
            (Expr::Cast { expr: ae, .. }, Expr::Cast { expr: be, .. }) => 1 + mcis_size(ae, be),
            _ => 1,
        }
    } else {
        // Roots differ — find best match among child pairs
        child_best_match(a, b)
    }
}

/// Check if two expression roots have the same structural type.
fn same_structure_root(a: &Expr, b: &Expr) -> bool {
    matches!(
        (a, b),
        (Expr::Literal(_), Expr::Literal(_))
            | (Expr::Column(_), Expr::Column(_))
            | (Expr::UnaryExpr { .. }, Expr::UnaryExpr { .. })
            | (Expr::BinaryExpr { .. }, Expr::BinaryExpr { .. })
            | (Expr::FunctionCall { .. }, Expr::FunctionCall { .. })
            | (Expr::Conditional { .. }, Expr::Conditional { .. })
            | (Expr::Aggregate { .. }, Expr::Aggregate { .. })
            | (Expr::Cast { .. }, Expr::Cast { .. })
    )
}

/// Best MCIS match among any pair of children from two expressions.
fn child_best_match(a: &Expr, b: &Expr) -> usize {
    let children_a = collect_children(a);
    let children_b = collect_children(b);
    if children_a.is_empty() || children_b.is_empty() {
        return 0;
    }
    let mut best = 0;
    for ca in &children_a {
        for cb in &children_b {
            let s = mcis_size(ca, cb);
            if s > best {
                best = s;
            }
        }
    }
    best
}

/// Collect direct children of an expression node.
fn collect_children(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::UnaryExpr { expr, .. } => vec![(**expr).clone()],
        Expr::BinaryExpr { left, right, .. } => vec![(**left).clone(), (**right).clone()],
        Expr::FunctionCall { args, .. } => args.clone(),
        Expr::Aggregate { expr, .. } => vec![(**expr).clone()],
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => vec![
            (**condition).clone(),
            (**then_expr).clone(),
            (**else_expr).clone(),
        ],
        Expr::Cast { expr, .. } => vec![(**expr).clone()],
        _ => vec![],
    }
}

/// Count total nodes in an expression.
fn expr_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Literal(_) | Expr::Column(_) => 1,
        Expr::UnaryExpr { expr, .. } => 1 + expr_node_count(expr),
        Expr::BinaryExpr { left, right, .. } => 1 + expr_node_count(left) + expr_node_count(right),
        Expr::FunctionCall { args, .. } => {
            1 + args.iter().map(|a| expr_node_count(a)).sum::<usize>()
        }
        Expr::Aggregate { expr, .. } => 1 + expr_node_count(expr),
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            1 + expr_node_count(condition) + expr_node_count(then_expr) + expr_node_count(else_expr)
        }
        Expr::Cast { expr, .. } => 1 + expr_node_count(expr),
    }
}

/// Structural similarity between two expressions (0.0–1.0).
/// Ratio of maximum common isomorphic subtree size to the larger tree size.
pub fn expr_structural_similarity(a: &Expr, b: &Expr) -> f64 {
    let mcis = mcis_size(a, b);
    let na = expr_node_count(a);
    let nb = expr_node_count(b);
    let max_n = na.max(nb);
    if max_n == 0 {
        0.0
    } else {
        mcis as f64 / max_n as f64
    }
}

/// Check a candidate expression against a pool for structural redundancy.
/// Returns the maximum similarity found, or None if the pool is empty.
pub fn check_redundancy(candidate: &Expr, pool: &[Expr]) -> Option<f64> {
    if pool.is_empty() {
        return None;
    }
    let max_sim = pool
        .iter()
        .map(|e| expr_structural_similarity(candidate, e))
        .fold(0.0_f64, f64::max);
    Some(max_sim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Literal;

    #[test]
    fn test_tree_operations() {
        // Create a simple expression: (x + 1) * 2
        let expr = Expr::Column("x".to_string())
            .add(Expr::Literal(Literal::Float(1.0)))
            .mul(Expr::Literal(Literal::Float(2.0)));

        // Test path collection
        let mut paths = Vec::new();
        tree_ops::collect_paths(&expr, &mut Vec::new(), &mut paths);
        assert!(!paths.is_empty());

        // Test get node at path
        if let Some(node) = tree_ops::get_node_at_path(&expr, &[0]) {
            assert!(matches!(node, Expr::BinaryExpr { .. }));
        }
    }
}
