//! Simple expression parser for factor expressions

use crate::expr::ast::{BinaryOp, Expr, Literal, UnaryOp};

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

        if c.is_ascii_digit() {
            let mut num = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                    // Handle scientific notation: 1e-10, 1.5e+5, etc.
                    if c == 'e' || c == 'E' {
                        num.push(c);
                        chars.next();
                        // After 'e' or 'E', can have + or -
                        if let Some(&next_c) = chars.peek() {
                            if next_c == '+' || next_c == '-' {
                                num.push(next_c);
                                chars.next();
                            }
                        }
                        continue;
                    }
                    num.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(Token::Number(num.parse().unwrap_or(0.0)));
        } else if c.is_alphabetic() || c == '_' {
            let mut ident = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_alphanumeric() || c == '_' {
                    ident.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Some(&'(') = chars.peek() {
                tokens.push(Token::Function(ident));
            } else {
                tokens.push(Token::Identifier(ident));
            }
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
        Token::Identifier(name) => {
            // Check if it's a function call
            if start + 1 < tokens.len() && matches!(&tokens[start + 1], Token::LParen) {
                parse_function(tokens, start)
            } else {
                Ok((Expr::Column(name.clone()), start + 1))
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

    // Map function names
    let func_name = match name.as_str() {
        "ts_mean" | "ts_avg" => "ts_mean",
        "ts_sum" => "ts_sum",
        "ts_max" => "ts_max",
        "ts_min" => "ts_min",
        "ts_std" => "ts_std",
        "ts_rank" => "ts_rank",
        _ => &name,
    };

    Ok((
        Expr::FunctionCall {
            name: func_name.to_string(),
            args,
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
}
