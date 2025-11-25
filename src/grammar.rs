//! Grammar-based text generation for Mullama
//!
//! This module provides comprehensive support for grammar-constrained generation,
//! allowing precise control over output format and structure.

use crate::error::MullamaError;
use crate::sys;
use std::ffi::CString;
use std::collections::HashMap;
use std::fmt;

/// Grammar definition for constrained generation
#[derive(Debug, Clone)]
pub struct Grammar {
    rules: HashMap<String, GrammarRule>,
    root_rule: String,
    compiled: Option<CompiledGrammar>,
}

/// A single grammar rule
#[derive(Debug, Clone)]
pub struct GrammarRule {
    pub name: String,
    pub alternatives: Vec<GrammarSequence>,
}

/// A sequence of grammar elements
#[derive(Debug, Clone)]
pub struct GrammarSequence {
    pub elements: Vec<GrammarElement>,
}

/// Individual grammar element
#[derive(Debug, Clone)]
pub enum GrammarElement {
    /// Terminal symbol (literal text)
    Terminal(String),
    /// Non-terminal symbol (reference to another rule)
    NonTerminal(String),
    /// Character class (e.g., [a-z])
    CharClass(CharClass),
    /// Optional element
    Optional(Box<GrammarElement>),
    /// Zero or more repetitions
    ZeroOrMore(Box<GrammarElement>),
    /// One or more repetitions
    OneOrMore(Box<GrammarElement>),
    /// Exact number of repetitions
    Repeat(Box<GrammarElement>, usize),
    /// Range of repetitions
    RepeatRange(Box<GrammarElement>, usize, usize),
}

/// Character class definition
#[derive(Debug, Clone)]
pub struct CharClass {
    pub ranges: Vec<(char, char)>,
    pub chars: Vec<char>,
    pub negated: bool,
}

/// Compiled grammar for efficient processing
#[derive(Debug)]
pub struct CompiledGrammar {
    /// The grammar as a GBNF string
    gbnf_string: CString,
    /// The root rule name
    root_rule: CString,
}

impl Clone for CompiledGrammar {
    fn clone(&self) -> Self {
        Self {
            gbnf_string: self.gbnf_string.clone(),
            root_rule: self.root_rule.clone(),
        }
    }
}

impl Grammar {
    /// Create a new empty grammar
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            root_rule: "root".to_string(),
            compiled: None,
        }
    }

    /// Create a grammar from GBNF (Grammar Backus-Naur Form) string
    ///
    /// # Example
    /// ```rust
    /// use mullama::grammar::Grammar;
    ///
    /// let grammar = Grammar::from_gbnf(r#"
    ///     root ::= "Hello" " " name
    ///     name ::= [A-Z][a-z]+
    /// "#)?;
    /// ```
    pub fn from_gbnf(gbnf: &str) -> Result<Self, MullamaError> {
        let mut grammar = Self::new();
        grammar.parse_gbnf(gbnf)?;
        Ok(grammar)
    }

    /// Create a grammar from a file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, MullamaError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MullamaError::GrammarError(format!("Failed to read grammar file: {}", e)))?;
        Self::from_gbnf(&content)
    }

    /// Add a rule to the grammar
    pub fn add_rule(&mut self, name: String, rule: GrammarRule) {
        self.rules.insert(name, rule);
        self.compiled = None; // Invalidate compilation
    }

    /// Set the root rule
    pub fn set_root(&mut self, root: String) -> Result<(), MullamaError> {
        if !self.rules.contains_key(&root) {
            return Err(MullamaError::GrammarError(format!(
                "Root rule '{}' not found",
                root
            )));
        }
        self.root_rule = root;
        self.compiled = None;
        Ok(())
    }

    /// Get a rule by name
    pub fn get_rule(&self, name: &str) -> Option<&GrammarRule> {
        self.rules.get(name)
    }

    /// Get all rule names
    pub fn rule_names(&self) -> Vec<&String> {
        self.rules.keys().collect()
    }

    /// Validate the grammar
    pub fn validate(&self) -> Result<(), MullamaError> {
        // Check that root rule exists
        if !self.rules.contains_key(&self.root_rule) {
            return Err(MullamaError::GrammarError(format!(
                "Root rule '{}' not found",
                self.root_rule
            )));
        }

        // Check that all non-terminals reference existing rules
        for (rule_name, rule) in &self.rules {
            for alternative in &rule.alternatives {
                for element in &alternative.elements {
                    self.validate_element(element, rule_name)?;
                }
            }
        }

        // Check for circular dependencies
        self.check_circular_dependencies()?;

        Ok(())
    }

    /// Compile the grammar for efficient use
    pub fn compile(&mut self) -> Result<(), MullamaError> {
        self.validate()?;

        // Convert to GBNF format for llama.cpp
        let gbnf_string = self.to_gbnf();
        let c_grammar = CString::new(gbnf_string)
            .map_err(|_| MullamaError::GrammarError("Invalid grammar string".to_string()))?;
        let c_root = CString::new(self.root_rule.clone())
            .map_err(|_| MullamaError::GrammarError("Invalid root rule name".to_string()))?;

        let compiled = CompiledGrammar {
            gbnf_string: c_grammar,
            root_rule: c_root,
        };

        self.compiled = Some(compiled);
        Ok(())
    }

    /// Get compiled grammar (compile if needed)
    pub fn get_compiled(&mut self) -> Result<&CompiledGrammar, MullamaError> {
        if self.compiled.is_none() {
            self.compile()?;
        }
        Ok(self.compiled.as_ref().unwrap())
    }

    /// Convert grammar to GBNF string format
    pub fn to_gbnf(&self) -> String {
        let mut result = String::new();

        // Start with root rule
        if let Some(root_rule) = self.rules.get(&self.root_rule) {
            result.push_str(&format!("{} ::= {}\n", self.root_rule, self.rule_to_gbnf(root_rule)));
        }

        // Add other rules
        for (name, rule) in &self.rules {
            if name != &self.root_rule {
                result.push_str(&format!("{} ::= {}\n", name, self.rule_to_gbnf(rule)));
            }
        }

        result
    }

    /// Parse GBNF string into grammar rules
    fn parse_gbnf(&mut self, gbnf: &str) -> Result<(), MullamaError> {
        for line in gbnf.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((name, definition)) = line.split_once("::=") {
                let name = name.trim().to_string();
                let definition = definition.trim();

                let rule = self.parse_rule_definition(definition)?;
                self.rules.insert(name, rule);
            }
        }

        Ok(())
    }

    /// Parse a single rule definition
    fn parse_rule_definition(&self, definition: &str) -> Result<GrammarRule, MullamaError> {
        // Split by | for alternatives
        let alternatives: Result<Vec<_>, _> = definition
            .split('|')
            .map(|alt| self.parse_sequence(alt.trim()))
            .collect();

        Ok(GrammarRule {
            name: String::new(), // Will be set by caller
            alternatives: alternatives?,
        })
    }

    /// Parse a sequence of elements
    fn parse_sequence(&self, sequence: &str) -> Result<GrammarSequence, MullamaError> {
        let mut elements = Vec::new();
        let mut chars = sequence.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '"' => {
                    // String literal
                    let mut literal = String::new();
                    while let Some(ch) = chars.next() {
                        if ch == '"' {
                            break;
                        }
                        if ch == '\\' {
                            if let Some(escaped) = chars.next() {
                                match escaped {
                                    'n' => literal.push('\n'),
                                    't' => literal.push('\t'),
                                    'r' => literal.push('\r'),
                                    '\\' => literal.push('\\'),
                                    '"' => literal.push('"'),
                                    _ => {
                                        literal.push('\\');
                                        literal.push(escaped);
                                    }
                                }
                            }
                        } else {
                            literal.push(ch);
                        }
                    }
                    elements.push(GrammarElement::Terminal(literal));
                }
                '[' => {
                    // Character class
                    let char_class = self.parse_char_class(&mut chars)?;
                    elements.push(GrammarElement::CharClass(char_class));
                }
                ' ' | '\t' => {
                    // Skip whitespace
                    continue;
                }
                _ => {
                    // Non-terminal or other construct
                    let mut name = String::new();
                    name.push(ch);

                    while let Some(&next_ch) = chars.peek() {
                        if next_ch.is_alphanumeric() || next_ch == '_' {
                            name.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }

                    // Check for modifiers
                    if let Some(&modifier) = chars.peek() {
                        match modifier {
                            '?' => {
                                chars.next();
                                elements.push(GrammarElement::Optional(Box::new(
                                    GrammarElement::NonTerminal(name)
                                )));
                            }
                            '*' => {
                                chars.next();
                                elements.push(GrammarElement::ZeroOrMore(Box::new(
                                    GrammarElement::NonTerminal(name)
                                )));
                            }
                            '+' => {
                                chars.next();
                                elements.push(GrammarElement::OneOrMore(Box::new(
                                    GrammarElement::NonTerminal(name)
                                )));
                            }
                            _ => {
                                elements.push(GrammarElement::NonTerminal(name));
                            }
                        }
                    } else {
                        elements.push(GrammarElement::NonTerminal(name));
                    }
                }
            }
        }

        Ok(GrammarSequence { elements })
    }

    /// Parse character class [a-z], [A-Z], etc.
    fn parse_char_class(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<CharClass, MullamaError> {
        let mut ranges = Vec::new();
        let mut single_chars = Vec::new();
        let mut negated = false;

        // Check for negation
        if let Some(&'^') = chars.peek() {
            chars.next();
            negated = true;
        }

        while let Some(ch) = chars.next() {
            if ch == ']' {
                break;
            }

            if let Some(&'-') = chars.peek() {
                chars.next(); // consume '-'
                if let Some(end_ch) = chars.next() {
                    if end_ch != ']' {
                        ranges.push((ch, end_ch));
                    } else {
                        // '-' at end, treat as literal
                        single_chars.push(ch);
                        single_chars.push('-');
                        break;
                    }
                }
            } else {
                single_chars.push(ch);
            }
        }

        Ok(CharClass {
            ranges,
            chars: single_chars,
            negated,
        })
    }

    /// Convert rule to GBNF format
    fn rule_to_gbnf(&self, rule: &GrammarRule) -> String {
        rule.alternatives
            .iter()
            .map(|alt| self.sequence_to_gbnf(alt))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Convert sequence to GBNF format
    fn sequence_to_gbnf(&self, sequence: &GrammarSequence) -> String {
        sequence.elements
            .iter()
            .map(|elem| self.element_to_gbnf(elem))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Convert element to GBNF format
    fn element_to_gbnf(&self, element: &GrammarElement) -> String {
        match element {
            GrammarElement::Terminal(s) => format!("\"{}\"", s),
            GrammarElement::NonTerminal(name) => name.clone(),
            GrammarElement::CharClass(class) => self.char_class_to_gbnf(class),
            GrammarElement::Optional(elem) => format!("({})?", self.element_to_gbnf(elem)),
            GrammarElement::ZeroOrMore(elem) => format!("({})*", self.element_to_gbnf(elem)),
            GrammarElement::OneOrMore(elem) => format!("({})+", self.element_to_gbnf(elem)),
            GrammarElement::Repeat(elem, count) => {
                format!("({}){{{}}}", self.element_to_gbnf(elem), count)
            }
            GrammarElement::RepeatRange(elem, min, max) => {
                format!("({}){{{},{}}}", self.element_to_gbnf(elem), min, max)
            }
        }
    }

    /// Convert character class to GBNF format
    fn char_class_to_gbnf(&self, class: &CharClass) -> String {
        let mut result = String::from("[");

        if class.negated {
            result.push('^');
        }

        for (start, end) in &class.ranges {
            result.push(*start);
            result.push('-');
            result.push(*end);
        }

        for ch in &class.chars {
            result.push(*ch);
        }

        result.push(']');
        result
    }

    /// Validate a grammar element
    fn validate_element(&self, element: &GrammarElement, context: &str) -> Result<(), MullamaError> {
        match element {
            GrammarElement::NonTerminal(name) => {
                if !self.rules.contains_key(name) {
                    return Err(MullamaError::GrammarError(format!(
                        "Rule '{}' references undefined rule '{}' in rule '{}'",
                        context, name, context
                    )));
                }
            }
            GrammarElement::Optional(elem) |
            GrammarElement::ZeroOrMore(elem) |
            GrammarElement::OneOrMore(elem) |
            GrammarElement::Repeat(elem, _) |
            GrammarElement::RepeatRange(elem, _, _) => {
                self.validate_element(elem, context)?;
            }
            _ => {} // Terminals and char classes are always valid
        }
        Ok(())
    }

    /// Check for circular dependencies in rules
    fn check_circular_dependencies(&self) -> Result<(), MullamaError> {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for rule_name in self.rules.keys() {
            if self.has_cycle(rule_name, &mut visited, &mut rec_stack)? {
                return Err(MullamaError::GrammarError(format!(
                    "Circular dependency detected involving rule '{}'",
                    rule_name
                )));
            }
        }

        Ok(())
    }

    /// Check if a rule has circular dependencies
    fn has_cycle(
        &self,
        rule_name: &str,
        visited: &mut std::collections::HashSet<String>,
        rec_stack: &mut std::collections::HashSet<String>,
    ) -> Result<bool, MullamaError> {
        if rec_stack.contains(rule_name) {
            return Ok(true);
        }

        if visited.contains(rule_name) {
            return Ok(false);
        }

        visited.insert(rule_name.to_string());
        rec_stack.insert(rule_name.to_string());

        if let Some(rule) = self.rules.get(rule_name) {
            for alternative in &rule.alternatives {
                for element in &alternative.elements {
                    if let Some(referenced_rule) = self.get_referenced_rule(element) {
                        if self.has_cycle(&referenced_rule, visited, rec_stack)? {
                            return Ok(true);
                        }
                    }
                }
            }
        }

        rec_stack.remove(rule_name);
        Ok(false)
    }

    /// Get the rule name referenced by an element
    fn get_referenced_rule(&self, element: &GrammarElement) -> Option<String> {
        match element {
            GrammarElement::NonTerminal(name) => Some(name.clone()),
            GrammarElement::Optional(elem) |
            GrammarElement::ZeroOrMore(elem) |
            GrammarElement::OneOrMore(elem) |
            GrammarElement::Repeat(elem, _) |
            GrammarElement::RepeatRange(elem, _, _) => self.get_referenced_rule(elem),
            _ => None,
        }
    }
}

impl Default for Grammar {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Grammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_gbnf())
    }
}

impl CompiledGrammar {
    /// Get the grammar string for use with llama.cpp
    pub fn grammar_str(&self) -> &CString {
        &self.gbnf_string
    }

    /// Get the root rule name
    pub fn root_rule(&self) -> &CString {
        &self.root_rule
    }

    /// Create a grammar sampler for this grammar
    ///
    /// # Arguments
    /// * `vocab` - The vocabulary to use (from model)
    ///
    /// # Safety
    /// The vocab pointer must be valid for the lifetime of the sampler
    pub fn create_sampler(&self, vocab: *const sys::llama_vocab) -> *mut sys::llama_sampler {
        unsafe {
            sys::llama_sampler_init_grammar(
                vocab,
                self.gbnf_string.as_ptr(),
                self.root_rule.as_ptr(),
            )
        }
    }

    /// Create a lazy grammar sampler that only activates on trigger words
    ///
    /// # Arguments
    /// * `model` - The model pointer
    /// * `trigger_words` - Words that trigger grammar enforcement
    /// * `trigger_tokens` - Tokens that trigger grammar enforcement
    pub fn create_lazy_sampler(
        &self,
        model: *const sys::llama_model,
        trigger_words: &[&str],
        trigger_tokens: &[i32],
    ) -> Result<*mut sys::llama_sampler, MullamaError> {
        // Convert trigger words to CStrings
        let c_trigger_words: Vec<CString> = trigger_words
            .iter()
            .map(|s| CString::new(*s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| MullamaError::GrammarError("Invalid trigger word".to_string()))?;

        let c_trigger_ptrs: Vec<*const std::os::raw::c_char> = c_trigger_words
            .iter()
            .map(|s| s.as_ptr())
            .collect();

        let sampler = unsafe {
            sys::llama_sampler_init_grammar_lazy(
                model,
                self.gbnf_string.as_ptr(),
                self.root_rule.as_ptr(),
                c_trigger_ptrs.as_ptr(),
                c_trigger_ptrs.len(),
                trigger_tokens.as_ptr(),
                trigger_tokens.len(),
            )
        };

        Ok(sampler)
    }
}

/// Grammar sampler wrapper for safe usage
pub struct GrammarSampler {
    sampler_ptr: *mut sys::llama_sampler,
}

impl GrammarSampler {
    /// Create a new grammar sampler from a compiled grammar
    pub fn new(grammar: &CompiledGrammar, vocab: *const sys::llama_vocab) -> Self {
        Self {
            sampler_ptr: grammar.create_sampler(vocab),
        }
    }

    /// Create from a GBNF string directly
    pub fn from_gbnf(
        vocab: *const sys::llama_vocab,
        gbnf: &str,
        root: &str,
    ) -> Result<Self, MullamaError> {
        let c_grammar = CString::new(gbnf)
            .map_err(|_| MullamaError::GrammarError("Invalid grammar string".to_string()))?;
        let c_root = CString::new(root)
            .map_err(|_| MullamaError::GrammarError("Invalid root rule".to_string()))?;

        let sampler_ptr = unsafe {
            sys::llama_sampler_init_grammar(
                vocab,
                c_grammar.as_ptr(),
                c_root.as_ptr(),
            )
        };

        Ok(Self { sampler_ptr })
    }

    /// Get the raw sampler pointer for use with llama.cpp
    pub fn as_ptr(&self) -> *mut sys::llama_sampler {
        self.sampler_ptr
    }

    /// Accept a token (call after sampling)
    pub fn accept(&mut self, token: i32) {
        unsafe {
            sys::llama_sampler_accept(self.sampler_ptr, token);
        }
    }

    /// Reset the sampler state
    pub fn reset(&mut self) {
        unsafe {
            sys::llama_sampler_reset(self.sampler_ptr);
        }
    }
}

impl Drop for GrammarSampler {
    fn drop(&mut self) {
        if !self.sampler_ptr.is_null() {
            unsafe {
                sys::llama_sampler_free(self.sampler_ptr);
            }
        }
    }
}

/// Predefined grammars for common formats
pub mod presets {
    use super::*;

    /// JSON grammar
    pub fn json() -> Result<Grammar, MullamaError> {
        Grammar::from_gbnf(r#"
            root ::= object
            value ::= object | array | string | number | boolean | null
            object ::= "{" "}"
            array ::= "[" "]"
            string ::= "\"" "\""
            number ::= [0-9]
            boolean ::= "true" | "false"
            null ::= "null"
        "#)
    }

    /// XML grammar
    pub fn xml() -> Result<Grammar, MullamaError> {
        Grammar::from_gbnf(r#"
            root ::= element
            element ::= "<" name attributes? ">" content? "</" name ">" | "<" name attributes? "/>"
            name ::= [a-zA-Z_] [a-zA-Z0-9_-]*
            attributes ::= (" " attribute)*
            attribute ::= name "=" "\"" [^"]* "\""
            content ::= (element | text)*
            text ::= [^<]+
        "#)
    }

    /// Simple programming language grammar
    pub fn simple_code() -> Result<Grammar, MullamaError> {
        Grammar::from_gbnf(r#"
            root ::= program
            program ::= (statement "\n")*
            statement ::= assignment | if_stmt | while_stmt | expression
            assignment ::= identifier " = " expression
            if_stmt ::= "if " expression " then\n" program "end"
            while_stmt ::= "while " expression " do\n" program "end"
            expression ::= term ((" + " | " - ") term)*
            term ::= factor ((" * " | " / ") factor)*
            factor ::= number | identifier | "(" expression ")"
            identifier ::= [a-zA-Z] [a-zA-Z0-9]*
            number ::= [0-9]+
        "#)
    }

    /// Email address grammar
    pub fn email() -> Result<Grammar, MullamaError> {
        Grammar::from_gbnf(r#"
            root ::= local "@" domain
            local ::= [a-zA-Z0-9._-]+
            domain ::= subdomain ("." subdomain)*
            subdomain ::= [a-zA-Z0-9-]+
        "#)
    }

    /// URL grammar
    pub fn url() -> Result<Grammar, MullamaError> {
        Grammar::from_gbnf(r##"
            root ::= scheme "://" authority path? query? fragment?
            scheme ::= "http" "s"?
            authority ::= host (":" port)?
            host ::= [a-zA-Z0-9.-]+
            port ::= [0-9]+
            path ::= ("/" [a-zA-Z0-9._-]*)*
            query ::= "?" [a-zA-Z0-9=&_-]*
            fragment ::= "#" [a-zA-Z0-9_-]*
        "##)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_grammar() {
        let grammar = Grammar::new();
        assert_eq!(grammar.rules.len(), 0);
        assert_eq!(grammar.root_rule, "root");
    }

    #[test]
    fn test_simple_gbnf() {
        let grammar = Grammar::from_gbnf(r#"
            root ::= "Hello" " " name
            name ::= [A-Z] [a-z]+
        "#).unwrap();

        assert_eq!(grammar.rules.len(), 2);
        assert!(grammar.rules.contains_key("root"));
        assert!(grammar.rules.contains_key("name"));
    }

    #[test]
    fn test_json_preset() {
        let grammar = presets::json().unwrap();
        grammar.validate().unwrap();
    }

    #[test]
    fn test_grammar_validation() {
        let mut grammar = Grammar::new();

        // Add a rule that references a non-existent rule
        let rule = GrammarRule {
            name: "test".to_string(),
            alternatives: vec![
                GrammarSequence {
                    elements: vec![GrammarElement::NonTerminal("nonexistent".to_string())],
                }
            ],
        };

        grammar.add_rule("test".to_string(), rule);
        grammar.set_root("test".to_string()).unwrap();

        // Validation should fail
        assert!(grammar.validate().is_err());
    }

    #[test]
    fn test_char_class() {
        let grammar = Grammar::from_gbnf(r#"
            root ::= [a-zA-Z0-9]
        "#).unwrap();

        if let Some(rule) = grammar.get_rule("root") {
            if let Some(alt) = rule.alternatives.first() {
                if let Some(GrammarElement::CharClass(class)) = alt.elements.first() {
                    assert_eq!(class.ranges.len(), 3); // a-z, A-Z, 0-9
                    assert!(!class.negated);
                }
            }
        }
    }

    #[test]
    fn test_grammar_to_gbnf() {
        let original_gbnf = r#"root ::= "Hello" " " name
name ::= [A-Z] [a-z]+"#;

        let grammar = Grammar::from_gbnf(original_gbnf).unwrap();
        let generated_gbnf = grammar.to_gbnf();

        // Parse the generated GBNF to ensure it's valid
        let reparsed = Grammar::from_gbnf(&generated_gbnf).unwrap();
        assert_eq!(reparsed.rules.len(), grammar.rules.len());
    }
}