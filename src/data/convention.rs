pub trait Convention {
    fn name(&self) -> &str;
}

pub enum DataType {
    Float64,
    Float32,
    Int64,
    Int32,
    VarChar,
    Datetime,
}

pub enum TableType {
    Stock1Min,
    Stock5Min,
    Stock1Day,
}

impl Convention for TableType {
    fn name(&self) -> &str {
        match self {
            TableType::Stock1Min => "stock_1min",
            TableType::Stock5Min => "stock_5min",
            TableType::Stock1Day => "stock_1day",
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Field {
    Symbol,
    TradingDay,
    Open,
    High,
    Low,
    Close,
    Volume,
    Amount,
    LimitUp,
    LimitDown,
    AdjustFactor,
    Pe,
    Pb,
}

impl Convention for Field {
    fn name(&self) -> &str {
        match self {
            Field::Symbol => "symbol",
            Field::TradingDay => "trading_day",
            Field::Open => "open",
            Field::High => "high",
            Field::Low => "low",
            Field::Close => "close",
            Field::Volume => "volume",
            Field::Amount => "amount",
            Field::LimitUp => "limit_up",
            Field::LimitDown => "limit_down",
            Field::AdjustFactor => "adjust_factor",
            Field::Pe => "pe",
            Field::Pb => "pb",
        }
    }
}
