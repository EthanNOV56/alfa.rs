//! Simple expression parser for factor expressions

use crate::expr::ast::{BinaryOp, Expr, Frequency, Literal, UnaryOp};

/// Parse expression string into Expr AST
pub fn parse_expression(expression: &str) -> Result<Expr, String> {
    let tokens = tokenize(expression)?;
    parse_tokens(&tokens)
}

fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        // Check for frequency prefix (5m:, 1d:, 1m:) or identifier
        if c.is_ascii_alphabetic() || c == '_' {
            let mut prefix = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_alphanumeric() || c == '_' || c == ':' {
                    prefix.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            // Check if it's a frequency prefix (ends with ':')
            if prefix.ends_with(':') && (prefix.starts_with("5m") || prefix.starts_with("1d") || prefix.starts_with("1m")) {
                tokens.push(Token::Frequency(prefix.trim_end_matches(':').to_string()));
                continue;
            }
            // Not a frequency prefix, process as identifier
            if let Some(&'(') = chars.peek() {
                tokens.push(Token::Function(prefix));
            } else {
                tokens.push(Token::Identifier(prefix));
            }
            continue;
        }

        if c.is_ascii_digit() {
            let mut num = String::new();
            let mut chars_clone = chars.clone();

            // Collect digits, optional decimal part, and optional scientific notation
            while let Some(&c) = chars_clone.peek() {
                if c.is_ascii_digit() {
                    num.push(c);
                    chars_clone.next();
                } else {
                    break;
                }
            }
            // Decimal part
            if chars_clone.peek() == Some(&'.') {
                num.push('.');
                chars_clone.next();
                while let Some(&c) = chars_clone.peek() {
                    if c.is_ascii_digit() {
                        num.push(c);
                        chars_clone.next();
                    } else {
                        break;
                    }
                }
            }
            // Scientific notation
            if chars_clone.peek() == Some(&'e') || chars_clone.peek() == Some(&'E') {
                num.push(chars_clone.next().unwrap());
                if chars_clone.peek() == Some(&'+') || chars_clone.peek() == Some(&'-') {
                    num.push(chars_clone.next().unwrap());
                }
                while let Some(&c) = chars_clone.peek() {
                    if c.is_ascii_digit() {
                        num.push(c);
                        chars_clone.next();
                    } else {
                        break;
                    }
                }
            }

            // Check for frequency suffix after number (e.g., "1d:", "5m:", "1m:")
            // Only trigger when num is a plain integer (no dot, no 'e')
            // The pattern is: integer followed by (d|m|h|w) followed by ':'
            if let Some(&c) = chars_clone.peek() {
                // c is the character after the digits (e.g., 'm' after '5')
                if c == 'd' || c == 'm' || c == 'h' || c == 'w' {
                    // Advance chars_clone to check the next character
                    chars_clone.next();
                    if let Some(&next_c) = chars_clone.peek() {
                        if next_c == ':' {
                            // This is a frequency suffix! e.g., "5m:" or "1d:"
                            // Consume 'd' or 'm' etc. and ':'
                            chars_clone.next(); // consume ':'
                            // Now chars_clone points to what follows the frequency suffix
                            // It could be a function name (e.g., "sum" in "1d:sum") or column name (e.g., "vol" in "5m:vol")
                            let mut after_freq = String::new();
                            while let Some(&fc) = chars_clone.peek() {
                                if fc.is_ascii_alphabetic() || fc == '_' {
                                    after_freq.push(fc);
                                    chars_clone.next();
                                } else {
                                    break;
                                }
                            }
                            // Check if after_freq is a known aggregation function
                            let valid_funcs = ["sum", "mean", "std", "min", "max", "count", "product"];
                            if valid_funcs.contains(&after_freq.as_str()) {
                                // This is frequency+function, like "1d:sum"
                                // Advance the REAL chars iterator to match what we consumed
                                for _ in 0..num.len() { chars.next(); } // consume digits
                                chars.next(); // consume 'd' or 'm' etc.
                                chars.next(); // consume ':'
                                for _ in 0..after_freq.len() { chars.next(); } // consume function name
                                let freq_str = format!("{}{}", num, c);
                                tokens.push(Token::Frequency(freq_str));
                                tokens.push(Token::Function(after_freq));
                                continue;
                            } else if !after_freq.is_empty() {
                                // It's frequency + column name, like "5m:vol"
                                // after_freq contains the column name
                                for _ in 0..num.len() { chars.next(); }
                                chars.next(); // consume 'd' or 'm' etc.
                                chars.next(); // consume ':'
                                for _ in 0..after_freq.len() { chars.next(); }
                                let freq_str = format!("{}{}", num, c);
                                tokens.push(Token::Frequency(freq_str));
                                tokens.push(Token::Identifier(after_freq));
                                continue;
                            }
                        }
                    }
                }
            }

            // Not a frequency suffix, just a number
            for _ in 0..num.len() { chars.next(); }
            tokens.push(Token::Number(num.parse().unwrap_or(0.0)));
        } else {
            let op = match c {
                '+' => Token::Plus,
                '-' => Token::Minus,
                '*' => Token::Multiply,
                '/' => Token::Divide,
                '(' => Token::LParen,
                ')' => Token::RParen,
                ',' => Token::Comma,
                _ => {
                    chars.next();
                    continue;
                }
            };
            tokens.push(op);
            chars.next();
        }
    }

    if tokens.is_empty() {
        return Err("Empty expression".to_string());
    }

    Ok(tokens)
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Identifier(String),
    Function(String),
    Frequency(String),  // "5m", "1d", "1m" etc.
    Plus,
    Minus,
    Multiply,
    Divide,
    LParen,
    RParen,
    Comma,
}

/// Parse tokens using a simple iterative shunting-yard inspired approach
fn parse_tokens(tokens: &[Token]) -> Result<Expr, String> {
    // Use recursive descent but with careful position tracking
    parse_expression_rec(tokens, 0).map(|(e, _)| e)
}

fn parse_expression_rec(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    if tokens.is_empty() {
        return Err("Empty tokens".to_string());
    }
    parse_additive(tokens, start)
}

fn parse_additive(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    let (mut left, mut pos) = parse_multiplicative(tokens, start)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Plus => {
                let (right, new_pos) = parse_multiplicative(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Add, right);
                pos = new_pos;
            }
            Token::Minus => {
                let (right, new_pos) = parse_multiplicative(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Subtract, right);
                pos = new_pos;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_multiplicative(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    let (mut left, mut pos) = parse_unary(tokens, start)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Multiply => {
                let (right, new_pos) = parse_unary(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Multiply, right);
                pos = new_pos;
            }
            Token::Divide => {
                let (right, new_pos) = parse_unary(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Divide, right);
                pos = new_pos;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_unary(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    if start >= tokens.len() {
        return Err("Unexpected end".to_string());
    }

    match &tokens[start] {
        Token::Minus => {
            let (expr, pos) = parse_primary(tokens, start + 1)?;
            Ok((expr.unary(UnaryOp::Negate), pos))
        }
        _ => parse_primary(tokens, start),
    }
}

fn parse_primary(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    if start >= tokens.len() {
        return Err("Unexpected end".to_string());
    }

    match &tokens[start] {
        Token::Number(n) => Ok((Expr::Literal(Literal::Float(*n)), start + 1)),
        Token::Frequency(freq) => {
            // Handle frequency prefix: 5m:vol or 1d:sum(...)
            if start + 1 >= tokens.len() {
                return Err("Unexpected end after frequency prefix".to_string());
            }
            match &tokens[start + 1] {
                Token::Identifier(name) => {
                    // 5m:vol -> Column with frequency prefix
                    Ok((Expr::Column(format!("{}:{}", freq, name)), start + 2))
                }
                Token::Function(_name) => {
                    // 1d:sum(...) -> Function with frequency prefix
                    let (expr, pos) = parse_function(tokens, start + 1)?;
                    // Attach the frequency to the FunctionCall
                    let expr = if let Expr::FunctionCall { name, args, freq: _ } = expr {
                        Expr::FunctionCall {
                            name,
                            args,
                            freq: Frequency::parse(freq),
                        }
                    } else {
                        expr
                    };
                    Ok((expr, pos))
                }
                _ => Err(format!(
                    "Expected identifier or function after frequency prefix, got {:?}",
                    tokens[start + 1]
                )),
            }
        }
        Token::Identifier(name) => {
            // Check if it's a function call
            if start + 1 < tokens.len() && matches!(&tokens[start + 1], Token::LParen) {
                parse_function(tokens, start)
            } else {
                let col = if name.contains(':') {
                    name.clone()
                } else {
                    format!("1d:{}", name)
                };
                Ok((Expr::Column(col), start + 1))
            }
        }
        Token::Function(_name) => parse_function(tokens, start),
        Token::LParen => {
            // Parenthesized expression
            let (expr, pos) = parse_additive(tokens, start + 1)?;
            if pos < tokens.len() && matches!(&tokens[pos], Token::RParen) {
                Ok((expr, pos + 1))
            } else {
                Err("Expected ')'".to_string())
            }
        }
        _ => Err(format!(
            "Unexpected token at position {}: {:?}",
            start, tokens[start]
        )),
    }
}

fn parse_function(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    let name = match &tokens[start] {
        Token::Identifier(n) => n.clone(),
        Token::Function(n) => n.clone(),
        _ => return Err("Expected function name".to_string()),
    };

    // Find opening paren
    let mut paren_pos = start;
    while paren_pos < tokens.len() && !matches!(&tokens[paren_pos], Token::LParen) {
        paren_pos += 1;
    }
    if paren_pos >= tokens.len() {
        return Err("Expected '('".to_string());
    }

    // Parse arguments
    let mut args = Vec::new();
    let mut pos = paren_pos + 1;
    let mut _arg_count = 0;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::RParen => {
                pos += 1;
                break;
            }
            Token::Comma => {
                pos += 1;
                _arg_count += 1;
            }
            _ => {
                let (expr, new_pos) = parse_additive(tokens, pos)?;
                args.push(expr);
                pos = new_pos;
            }
        }
    }

    // Map function names (case-insensitive) to canonical names
    // Canonical naming: ts_ prefix for time-series, cs_ prefix for cross-sectional,
    // no prefix for element-wise/scalar functions.
    let name_lower = name.to_lowercase();
    let func_name = match name_lower.as_str() {
        // ── Cross-sectional functions (cs_ prefix) ──
        "cs_rank" => "cs_rank",
        "cs_scale" => "cs_scale",
        "rank" => "cs_rank",
        "scale" => "cs_scale",
        // ── Time-series rolling window (ts_ prefix) ──
        "ts_mean" | "ts_avg" => "ts_mean",
        "ts_sum" => "ts_sum",
        "ts_max" => "ts_max",
        "ts_min" => "ts_min",
        "ts_std" => "ts_std",
        "ts_rank" => "ts_rank",
        "ts_argmax" => "ts_argmax",
        "ts_argmin" => "ts_argmin",
        "ts_correlation" => "ts_correlation",
        "ts_covariance" | "ts_cov" => "ts_covariance",
        "ts_delta" => "ts_delta",
        "ts_product" => "ts_product",
        "ts_count" => "ts_count",
        // backward-compat: bare names normalize to canonical
        "mean" | "avg" => "ts_mean",
        "sum" => "ts_sum",
        "max" => "ts_max",
        "min" => "ts_min",
        "std" => "ts_std",
        "argmax" => "ts_argmax",
        "argmin" => "ts_argmin",
        "correlation" | "corr" => "ts_correlation",
        "covariance" | "cov" => "ts_covariance",
        "delta" => "ts_delta",
        "product" => "ts_product",
        "count" | "cnt" => "ts_count",
        // ── Time-series non-rolling (ts_ prefix, scalar impl) ──
        "ts_delay" => "ts_delay",
        "ts_decay_linear" => "ts_decay_linear",
        "ts_sma" => "ts_sma",
        "delay" => "ts_delay",
        "decay_linear" | "decay" => "ts_decay_linear",
        "sma" | "ema" => "ts_sma",
        // ── Time-series specialized (ts_ prefix) ──
        "ts_wma" => "ts_wma",
        "ts_lowday" => "ts_lowday",
        "ts_highday" => "ts_highday",
        "wma" => "ts_wma",
        "lowday" => "ts_lowday",
        "highday" => "ts_highday",
        // ── New time-series (ts_ prefix) ──
        "ts_quantile" => "ts_quantile",
        "ts_slope" => "ts_slope",
        "ts_rsquare" => "ts_rsquare",
        "ts_resi" => "ts_resi",
        // ── Element-wise / scalar (no prefix) ──
        "sign" => "sign",
        "abs" => "abs",
        "log" => "log",
        "log10" => "log10",
        "sqrt" => "sqrt",
        "power" => "power",
        "pow1" => "power",
        "pow2" => "power",
        // backward-compat prefixed forms for element-wise
        "ts_abs" => "abs",
        "ts_log" => "log",
        "ts_greater" => "gt",
        "ts_less" => "lt",
        // ── Conditional / comparison (no prefix) ──
        "if" => "if",
        "gt" | "greater" => "gt",
        "lt" | "less" => "lt",
        "ge" | "greater_equal" | "gte" => "ge",
        "le" | "less_equal" | "lte" => "le",
        "eq" | "equal" => "eq",
        "ne" | "not_equal" => "ne",
        // ── Custom conditional ──
        "quesval" => "quesval",
        "quesval2" => "quesval2",
        // ── Derived ──
        "returns" => "returns",
        _ => &name,
    };

    Ok((
        Expr::FunctionCall {
            name: func_name.to_string(),
            args,
            freq: None,
        },
        pos,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("close + volume").unwrap();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_parse_column() {
        let expr = parse_expression("close").unwrap();
        assert!(matches!(expr, Expr::Column(_)));
    }

    #[test]
    fn test_parse_binary() {
        let expr = parse_expression("close + volume").unwrap();
        assert!(matches!(expr, Expr::BinaryExpr { .. }));
    }

    #[test]
    fn test_parse_function() {
        let expr = parse_expression("ts_mean(close, 20)").unwrap();
        assert!(matches!(expr, Expr::FunctionCall { .. }));
    }

    #[test]
    fn test_parse_number() {
        // Numbers are parsed as floats (parser limitation)
        let expr = parse_expression("42").unwrap();
        assert!(matches!(expr, Expr::Literal(Literal::Float(f)) if (f - 42.0).abs() < 1e-10));

        // Test float
        let expr = parse_expression("3.14").unwrap();
        assert!(matches!(expr, Expr::Literal(Literal::Float(f)) if (f - 3.14).abs() < 1e-10));

        // Test negative number (parsed as unary minus)
        let expr = parse_expression("-5").unwrap();
        assert!(matches!(expr, Expr::UnaryExpr { .. }));

        // Test scientific notation
        let expr = parse_expression("1e-10").unwrap();
        assert!(matches!(expr, Expr::Literal(Literal::Float(f)) if (f - 1e-10).abs() < 1e-20));
    }

    #[test]
    fn test_parse_identifier() {
        let expr = parse_expression("close").unwrap();
        assert!(matches!(expr, Expr::Column(name) if name == "close"));

        let expr = parse_expression("volume_123").unwrap();
        assert!(matches!(expr, Expr::Column(name) if name == "volume_123"));

        let expr = parse_expression("_private").unwrap();
        assert!(matches!(expr, Expr::Column(name) if name == "_private"));
    }

    #[test]
    fn test_parse_nested_expression() {
        // Test nested function calls
        let expr = parse_expression("rank(ts_mean(close, 20))").unwrap();
        assert!(matches!(expr, Expr::FunctionCall { name, .. } if name == "rank"));

        // Test multiple operations
        let expr = parse_expression("close + open * 2").unwrap();
        assert!(matches!(expr, Expr::BinaryExpr { .. }));

        // Test parentheses (not currently supported but shouldn't crash)
        // Currently parentheses are not supported, so it will fail
    }

    #[test]
    fn test_parse_error() {
        // Test empty expression
        let result = parse_expression("");
        assert!(result.is_err());

        // Test invalid expression
        let result = parse_expression("close + ");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenize_complex() {
        let tokens = tokenize("rank(ts_mean(close, 20)) + volume / 100").unwrap();
        // Should have: rank, (, ts_mean, (, close, ,, 20, ), ), +, volume, /, 100
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_parse_1d_aggregation() {
        // Test the problematic expression
        let expr = parse_expression("1d:sum(5m:vol * 5m:close)");
        eprintln!("1d:sum(5m:vol * 5m:close) = {:?}", expr);
        assert!(expr.is_ok());

        let expr2 = parse_expression("1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)");
        eprintln!("full expr = {:?}", expr2);
        assert!(expr2.is_ok());
    }
}
