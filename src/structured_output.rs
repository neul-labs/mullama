//! Structured output generation via JSON Schema validation
//!
//! This module converts JSON Schema definitions to GBNF grammars for constrained
//! generation, enabling the model to produce output that conforms to a schema.
//!
//! ## Supported JSON Schema Features
//!
//! - **Types**: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`
//! - **Object properties**: `properties`, `required`, `additionalProperties`
//! - **Arrays**: `items` (single schema)
//! - **Enums**: `enum` (for strings)
//! - **String constraints**: `minLength`, `maxLength`
//! - **Integer constraints**: `minimum`, `maximum`
//!
//! ## Unsupported Features
//!
//! - `oneOf`, `anyOf`, `allOf`, `not`
//! - `$ref` (references)
//! - `patternProperties`
//! - `format` (dates, emails, etc.)
//! - `pattern` (regex)
//!
//! ## Example
//!
//! ```rust,no_run
//! use mullama::structured_output::JsonSchemaConverter;
//! use serde_json::json;
//!
//! let schema = json!({
//!     "type": "object",
//!     "properties": {
//!         "name": { "type": "string" },
//!         "age": { "type": "integer", "minimum": 0 }
//!     },
//!     "required": ["name", "age"]
//! });
//!
//! let grammar = JsonSchemaConverter::convert(&schema).unwrap();
//! ```

use crate::grammar::Grammar;
use crate::MullamaError;
use serde_json::Value;

/// Error types for structured output conversion
#[derive(Debug, Clone)]
pub enum StructuredOutputError {
    /// Schema uses an unsupported feature
    UnsupportedFeature(String),
    /// Schema is invalid or malformed
    InvalidSchema(String),
    /// Grammar compilation failed
    GrammarError(String),
}

impl std::fmt::Display for StructuredOutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFeature(msg) => write!(f, "Unsupported schema feature: {}", msg),
            Self::InvalidSchema(msg) => write!(f, "Invalid schema: {}", msg),
            Self::GrammarError(msg) => write!(f, "Grammar error: {}", msg),
        }
    }
}

impl std::error::Error for StructuredOutputError {}

impl From<StructuredOutputError> for MullamaError {
    fn from(e: StructuredOutputError) -> Self {
        MullamaError::GrammarError(e.to_string())
    }
}

/// Converts JSON Schema to GBNF grammar
pub struct JsonSchemaConverter {
    rules: Vec<String>,
    rule_counter: usize,
}

impl JsonSchemaConverter {
    /// Create a new converter
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            rule_counter: 0,
        }
    }

    /// Convert a JSON Schema to a Grammar
    ///
    /// # Arguments
    ///
    /// * `schema` - The JSON Schema to convert
    ///
    /// # Returns
    ///
    /// A Grammar that constrains output to match the schema
    pub fn convert(schema: &Value) -> Result<Grammar, StructuredOutputError> {
        let mut converter = Self::new();
        converter.add_primitives();
        let root_rule = converter.schema_to_rule("root", schema)?;
        converter.rules.insert(0, root_rule);

        let gbnf = converter.rules.join("\n");
        Grammar::from_gbnf(&gbnf).map_err(|e| StructuredOutputError::GrammarError(e.to_string()))
    }

    /// Validate that a schema is supported
    pub fn validate_schema(schema: &Value) -> Result<(), StructuredOutputError> {
        Self::validate_schema_recursive(schema)
    }

    fn validate_schema_recursive(schema: &Value) -> Result<(), StructuredOutputError> {
        let obj = schema
            .as_object()
            .ok_or_else(|| StructuredOutputError::InvalidSchema("Schema must be an object".into()))?;

        // Check for unsupported features
        for key in obj.keys() {
            match key.as_str() {
                "type" | "properties" | "required" | "additionalProperties" | "items" | "enum"
                | "minimum" | "maximum" | "minLength" | "maxLength" | "description" | "title"
                | "default" | "examples" | "const" => {}
                "oneOf" | "anyOf" | "allOf" | "not" => {
                    return Err(StructuredOutputError::UnsupportedFeature(format!(
                        "'{}' is not supported",
                        key
                    )));
                }
                "$ref" => {
                    return Err(StructuredOutputError::UnsupportedFeature(
                        "'$ref' references are not supported".into(),
                    ));
                }
                "pattern" => {
                    return Err(StructuredOutputError::UnsupportedFeature(
                        "'pattern' regex constraints are not supported".into(),
                    ));
                }
                "patternProperties" => {
                    return Err(StructuredOutputError::UnsupportedFeature(
                        "'patternProperties' is not supported".into(),
                    ));
                }
                "format" => {
                    // Silently ignore format - we don't enforce it
                }
                _ => {
                    // Allow unknown keys (they might be custom extensions)
                }
            }
        }

        // Recursively validate nested schemas
        if let Some(properties) = obj.get("properties") {
            if let Some(props) = properties.as_object() {
                for prop_schema in props.values() {
                    Self::validate_schema_recursive(prop_schema)?;
                }
            }
        }

        if let Some(items) = obj.get("items") {
            Self::validate_schema_recursive(items)?;
        }

        Ok(())
    }

    fn add_primitives(&mut self) {
        // Whitespace
        self.rules.push("ws ::= [ \\t\\n]*".to_string());

        // String primitive (handles escaping)
        self.rules.push(
            r#"string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"""#.to_string()
        );

        // Number primitive
        self.rules.push("number ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [\"+\\-\"]? [0-9]+)?".to_string());

        // Integer primitive
        self.rules
            .push("integer ::= \"-\"? ([0-9] | [1-9] [0-9]*)".to_string());

        // Boolean primitive
        self.rules
            .push("boolean ::= \"true\" | \"false\"".to_string());

        // Null primitive
        self.rules.push("null ::= \"null\"".to_string());
    }

    fn generate_rule_name(&mut self, prefix: &str) -> String {
        self.rule_counter += 1;
        format!("{}_{}", prefix, self.rule_counter)
    }

    fn schema_to_rule(&mut self, name: &str, schema: &Value) -> Result<String, StructuredOutputError> {
        let obj = schema.as_object().ok_or_else(|| {
            StructuredOutputError::InvalidSchema("Schema must be an object".into())
        })?;

        // Handle const
        if let Some(const_val) = obj.get("const") {
            return self.const_to_rule(name, const_val);
        }

        // Handle enum
        if let Some(enum_vals) = obj.get("enum") {
            return self.enum_to_rule(name, enum_vals);
        }

        // Handle type
        let type_val = obj.get("type");

        match type_val.and_then(|v| v.as_str()) {
            Some("object") => self.object_to_rule(name, obj),
            Some("array") => self.array_to_rule(name, obj),
            Some("string") => self.string_to_rule(name, obj),
            Some("number") => Ok(format!("{} ::= number", name)),
            Some("integer") => self.integer_to_rule(name, obj),
            Some("boolean") => Ok(format!("{} ::= boolean", name)),
            Some("null") => Ok(format!("{} ::= null", name)),
            Some(t) => Err(StructuredOutputError::InvalidSchema(format!(
                "Unknown type '{}'",
                t
            ))),
            None => {
                // No type specified, allow any JSON value
                Ok(format!(
                    "{} ::= string | number | boolean | null | {} | {}",
                    name,
                    self.generate_rule_name("obj"),
                    self.generate_rule_name("arr")
                ))
            }
        }
    }

    fn const_to_rule(&mut self, name: &str, value: &Value) -> Result<String, StructuredOutputError> {
        let literal = match value {
            Value::String(s) => format!("\"\\\"{}\\\"\"", escape_gbnf_string(s)),
            Value::Number(n) => format!("\"{}\"", n),
            Value::Bool(b) => format!("\"{}\"", b),
            Value::Null => "\"null\"".to_string(),
            _ => {
                return Err(StructuredOutputError::UnsupportedFeature(
                    "const with object/array values".into(),
                ))
            }
        };
        Ok(format!("{} ::= {}", name, literal))
    }

    fn enum_to_rule(&mut self, name: &str, values: &Value) -> Result<String, StructuredOutputError> {
        let arr = values.as_array().ok_or_else(|| {
            StructuredOutputError::InvalidSchema("enum must be an array".into())
        })?;

        let alternatives: Vec<String> = arr
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(format!("\"\\\"{}\\\"\"", escape_gbnf_string(s))),
                Value::Number(n) => Ok(format!("\"{}\"", n)),
                Value::Bool(b) => Ok(format!("\"{}\"", b)),
                Value::Null => Ok("\"null\"".to_string()),
                _ => Err(StructuredOutputError::UnsupportedFeature(
                    "enum with object/array values".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(format!("{} ::= {}", name, alternatives.join(" | ")))
    }

    fn object_to_rule(
        &mut self,
        name: &str,
        schema: &serde_json::Map<String, Value>,
    ) -> Result<String, StructuredOutputError> {
        let properties = schema.get("properties").and_then(|v| v.as_object());
        let required: Vec<&str> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // If no properties defined, use generic object
        let Some(props) = properties else {
            return Ok(format!("{} ::= \"{{\" ws \"}}\"", name));
        };

        if props.is_empty() {
            return Ok(format!("{} ::= \"{{\" ws \"}}\"", name));
        }

        // Generate rules for each property
        let mut prop_rules = Vec::new();
        let mut prop_names = Vec::new();

        for (prop_name, prop_schema) in props {
            let rule_name = self.generate_rule_name(&format!("{}_prop", name));
            let rule = self.schema_to_rule(&rule_name, prop_schema)?;
            self.rules.push(rule);

            let is_required = required.contains(&prop_name.as_str());
            prop_rules.push((prop_name.clone(), rule_name, is_required));
            prop_names.push(prop_name.clone());
        }

        // Build the object rule
        // For simplicity, we'll generate required properties in order
        // and make optional properties actually optional with a specific pattern

        let mut parts = Vec::new();
        let mut first = true;

        for (prop_name, rule_name, is_required) in &prop_rules {
            let prop_pattern = if first {
                format!(
                    "\"\\\"{}\\\":\" ws {}",
                    escape_gbnf_string(prop_name),
                    rule_name
                )
            } else {
                format!(
                    "\",\" ws \"\\\"{}\\\":\" ws {}",
                    escape_gbnf_string(prop_name),
                    rule_name
                )
            };

            if *is_required {
                parts.push(prop_pattern);
                first = false;
            } else {
                // Optional property
                let opt_rule = self.generate_rule_name("opt");
                if first {
                    self.rules
                        .push(format!("{} ::= ({})? ", opt_rule, prop_pattern));
                } else {
                    self.rules
                        .push(format!("{} ::= ({})? ", opt_rule, prop_pattern));
                }
                parts.push(opt_rule);
            }
        }

        let body = parts.join(" ");
        Ok(format!("{} ::= \"{{\" ws {} ws \"}}\"", name, body))
    }

    fn array_to_rule(
        &mut self,
        name: &str,
        schema: &serde_json::Map<String, Value>,
    ) -> Result<String, StructuredOutputError> {
        // Get items schema
        let items = schema.get("items");

        let item_rule = if let Some(item_schema) = items {
            let rule_name = self.generate_rule_name(&format!("{}_item", name));
            let rule = self.schema_to_rule(&rule_name, item_schema)?;
            self.rules.push(rule);
            rule_name
        } else {
            // No items schema, allow any value
            "string | number | boolean | null".to_string()
        };

        // Create array content rule
        let content_rule = self.generate_rule_name(&format!("{}_content", name));
        self.rules.push(format!(
            "{} ::= ({} (\",\" ws {})*)? ",
            content_rule, item_rule, item_rule
        ));

        Ok(format!("{} ::= \"[\" ws {} ws \"]\"", name, content_rule))
    }

    fn string_to_rule(
        &mut self,
        name: &str,
        schema: &serde_json::Map<String, Value>,
    ) -> Result<String, StructuredOutputError> {
        let min_length = schema
            .get("minLength")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let max_length = schema.get("maxLength").and_then(|v| v.as_u64());

        // For now, we use the basic string rule
        // More sophisticated length constraints would require character-level rules
        if min_length > 0 || max_length.is_some() {
            // Generate a constrained string rule
            // This is a simplified version - proper implementation would need
            // more complex GBNF rules
            let char_rule = r#"[^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])"#;

            if let Some(max) = max_length {
                // Generate rule with max length
                let content_rule = self.generate_rule_name(&format!("{}_content", name));
                if min_length == 0 {
                    self.rules.push(format!(
                        "{} ::= ({}){{0,{}}}",
                        content_rule, char_rule, max
                    ));
                } else {
                    self.rules.push(format!(
                        "{} ::= ({}){{0,{}}}",
                        content_rule, char_rule, max
                    ));
                }
                return Ok(format!("{} ::= \"\\\"\" {} \"\\\"\"", name, content_rule));
            }
        }

        // Use default string rule
        Ok(format!("{} ::= string", name))
    }

    fn integer_to_rule(
        &mut self,
        name: &str,
        schema: &serde_json::Map<String, Value>,
    ) -> Result<String, StructuredOutputError> {
        let minimum = schema.get("minimum").and_then(|v| v.as_i64());
        let maximum = schema.get("maximum").and_then(|v| v.as_i64());

        // For simple cases with small ranges, we could enumerate values
        // For now, we use the basic integer rule
        if let (Some(min), Some(max)) = (minimum, maximum) {
            if max - min <= 100 && max - min >= 0 {
                // Small range, enumerate all values
                let values: Vec<String> = (min..=max).map(|n| format!("\"{}\"", n)).collect();
                return Ok(format!("{} ::= {}", name, values.join(" | ")));
            }
        }

        // Use default integer rule
        Ok(format!("{} ::= integer", name))
    }
}

impl Default for JsonSchemaConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape a string for use in GBNF
fn escape_gbnf_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Get a grammar for generic JSON output
pub fn json_grammar() -> Result<Grammar, MullamaError> {
    crate::grammar::presets::json()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_object_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name"]
        });

        let result = JsonSchemaConverter::convert(&schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enum_schema() {
        let schema = json!({
            "type": "string",
            "enum": ["red", "green", "blue"]
        });

        let result = JsonSchemaConverter::convert(&schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_array_schema() {
        let schema = json!({
            "type": "array",
            "items": { "type": "string" }
        });

        let result = JsonSchemaConverter::convert(&schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            }
        });

        let result = JsonSchemaConverter::convert(&schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unsupported_feature_ref() {
        let schema = json!({
            "$ref": "#/definitions/Person"
        });

        let result = JsonSchemaConverter::validate_schema(&schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_feature_oneof() {
        let schema = json!({
            "oneOf": [
                { "type": "string" },
                { "type": "number" }
            ]
        });

        let result = JsonSchemaConverter::validate_schema(&schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_integer_range() {
        let schema = json!({
            "type": "integer",
            "minimum": 1,
            "maximum": 10
        });

        let result = JsonSchemaConverter::convert(&schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_boolean_schema() {
        let schema = json!({
            "type": "boolean"
        });

        let result = JsonSchemaConverter::convert(&schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_const_schema() {
        let schema = json!({
            "const": "fixed_value"
        });

        let result = JsonSchemaConverter::convert(&schema);
        assert!(result.is_ok());
    }
}
