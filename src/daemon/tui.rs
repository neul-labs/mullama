//! TUI (Terminal User Interface) for Mullama
//!
//! A polished chat interface with model selection, conversation history, and rich formatting.

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{
        Block, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Wrap,
    },
    Frame, Terminal,
};

use super::client::DaemonClient;
use super::protocol::{DaemonStatus, ModelStatus};

/// Chat message
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub model: Option<String>,
    pub tokens: Option<u32>,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
    System,
}

/// Input mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputMode {
    Normal,
    Insert,
    Command,
    ModelSelect,
}

/// TUI Application
pub struct TuiApp {
    client: DaemonClient,

    // State
    messages: Vec<Message>,
    input: String,
    cursor_pos: usize,
    input_mode: InputMode,

    // Scrolling
    messages_scroll: usize,
    messages_scroll_state: ScrollbarState,

    // Models
    models: Vec<ModelStatus>,
    selected_model: Option<String>,
    model_select_index: usize,

    // Status
    daemon_status: Option<DaemonStatus>,
    status_message: String,
    last_error: Option<String>,

    // Generation params
    max_tokens: u32,
    temperature: f32,

    // Flags
    generating: bool,
    should_quit: bool,
    show_help: bool,

    // History
    input_history: Vec<String>,
    history_index: Option<usize>,
}

impl TuiApp {
    pub fn new(client: DaemonClient) -> Self {
        Self {
            client,
            messages: vec![],
            input: String::new(),
            cursor_pos: 0,
            input_mode: InputMode::Insert,
            messages_scroll: 0,
            messages_scroll_state: ScrollbarState::default(),
            models: vec![],
            selected_model: None,
            model_select_index: 0,
            daemon_status: None,
            status_message: String::new(),
            last_error: None,
            max_tokens: 512,
            temperature: 0.7,
            generating: false,
            should_quit: false,
            show_help: false,
            input_history: vec![],
            history_index: None,
        }
    }

    /// Run the TUI
    pub fn run(&mut self) -> io::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Initial data fetch
        self.refresh_status();
        self.refresh_models();

        // Welcome message
        self.add_system_message("Welcome to Mullama! Type a message to chat, or press ? for help.");

        let result = self.event_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    fn event_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        loop {
            terminal.draw(|f| self.draw(f))?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key);
                }
            }

            if self.should_quit {
                break;
            }
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) {
        // Global shortcuts
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    if self.generating {
                        self.status_message = "Cancelled".to_string();
                        self.generating = false;
                    } else {
                        self.should_quit = true;
                    }
                    return;
                }
                KeyCode::Char('q') => {
                    self.should_quit = true;
                    return;
                }
                KeyCode::Char('l') => {
                    self.messages.clear();
                    self.add_system_message("Chat cleared.");
                    return;
                }
                KeyCode::Char('m') => {
                    self.input_mode = InputMode::ModelSelect;
                    self.model_select_index = 0;
                    return;
                }
                _ => {}
            }
        }

        // Help toggle
        if key.code == KeyCode::Char('?') && self.input_mode == InputMode::Normal {
            self.show_help = !self.show_help;
            return;
        }

        // Mode-specific handling
        match self.input_mode {
            InputMode::Insert => self.handle_insert_key(key),
            InputMode::Normal => self.handle_normal_key(key),
            InputMode::Command => self.handle_command_key(key),
            InputMode::ModelSelect => self.handle_model_select_key(key),
        }
    }

    fn handle_insert_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                if !self.input.is_empty() {
                    self.submit_input();
                }
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.input.len());
            }
            KeyCode::Home => self.cursor_pos = 0,
            KeyCode::End => self.cursor_pos = self.input.len(),
            KeyCode::Up => {
                if !self.input_history.is_empty() {
                    let idx = match self.history_index {
                        Some(i) => i.saturating_sub(1),
                        None => self.input_history.len() - 1,
                    };
                    self.history_index = Some(idx);
                    self.input = self.input_history[idx].clone();
                    self.cursor_pos = self.input.len();
                }
            }
            KeyCode::Down => {
                if let Some(idx) = self.history_index {
                    if idx + 1 < self.input_history.len() {
                        self.history_index = Some(idx + 1);
                        self.input = self.input_history[idx + 1].clone();
                    } else {
                        self.history_index = None;
                        self.input.clear();
                    }
                    self.cursor_pos = self.input.len();
                }
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Normal;
            }
            KeyCode::PageUp => {
                self.messages_scroll = self.messages_scroll.saturating_sub(5);
            }
            KeyCode::PageDown => {
                self.messages_scroll += 5;
            }
            _ => {}
        }
    }

    fn handle_normal_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('i') | KeyCode::Char('a') => {
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Char(':') => {
                self.input_mode = InputMode::Command;
                self.input.clear();
                self.cursor_pos = 0;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.messages_scroll += 1;
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.messages_scroll = self.messages_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') => {
                self.messages_scroll = 0;
            }
            KeyCode::Char('G') => {
                self.messages_scroll = self.messages.len().saturating_sub(1);
            }
            KeyCode::Char('q') => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn handle_command_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                self.execute_command();
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Esc => {
                self.input.clear();
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Char(c) => {
                self.input.push(c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if !self.input.is_empty() {
                    self.input.pop();
                    self.cursor_pos = self.cursor_pos.saturating_sub(1);
                } else {
                    self.input_mode = InputMode::Insert;
                }
            }
            _ => {}
        }
    }

    fn handle_model_select_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.model_select_index = self.model_select_index.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.models.is_empty() {
                    self.model_select_index =
                        (self.model_select_index + 1).min(self.models.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(model) = self.models.get(self.model_select_index) {
                    self.selected_model = Some(model.alias.clone());
                    self.status_message = format!("Selected model: {}", model.alias);
                }
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Esc => {
                self.input_mode = InputMode::Insert;
            }
            _ => {}
        }
    }

    fn submit_input(&mut self) {
        let input = std::mem::take(&mut self.input);
        self.cursor_pos = 0;
        self.history_index = None;

        // Add to history
        if !input.is_empty() {
            self.input_history.push(input.clone());
        }

        // Add user message
        self.messages.push(Message {
            role: Role::User,
            content: input.clone(),
            model: None,
            tokens: None,
            duration_ms: None,
        });

        // Scroll to bottom
        self.messages_scroll = self.messages.len();

        // Generate response
        self.generating = true;
        self.status_message = "Generating...".to_string();

        let model = self.selected_model.clone();
        match self
            .client
            .chat(&input, model.as_deref(), self.max_tokens, self.temperature)
        {
            Ok(result) => {
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: result.text,
                    model: Some(result.model),
                    tokens: Some(result.completion_tokens),
                    duration_ms: Some(result.duration_ms),
                });

                let tps = if result.duration_ms > 0 {
                    (result.completion_tokens as f64) / (result.duration_ms as f64 / 1000.0)
                } else {
                    0.0
                };
                self.status_message = format!(
                    "{} tokens in {}ms ({:.1} tok/s)",
                    result.completion_tokens, result.duration_ms, tps
                );
            }
            Err(e) => {
                self.last_error = Some(e.to_string());
                self.status_message = "Generation failed".to_string();
                self.add_system_message(&format!("Error: {}", e));
            }
        }

        self.generating = false;
        self.messages_scroll = self.messages.len();
    }

    fn execute_command(&mut self) {
        let cmd = std::mem::take(&mut self.input);
        let parts: Vec<&str> = cmd.split_whitespace().collect();

        match parts.first().map(|s| *s) {
            Some("q") | Some("quit") => {
                self.should_quit = true;
            }
            Some("clear") | Some("c") => {
                self.messages.clear();
                self.add_system_message("Chat cleared.");
            }
            Some("models") | Some("m") => {
                self.refresh_models();
                let list = self
                    .models
                    .iter()
                    .map(|m| format!("{}{}", m.alias, if m.is_default { " *" } else { "" }))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.add_system_message(&format!("Models: {}", list));
            }
            Some("model") => {
                if let Some(alias) = parts.get(1) {
                    self.selected_model = Some(alias.to_string());
                    self.status_message = format!("Selected: {}", alias);
                } else {
                    let current = self.selected_model.as_deref().unwrap_or("default");
                    self.add_system_message(&format!("Current model: {}", current));
                }
            }
            Some("load") => {
                if parts.len() >= 2 {
                    let spec = parts[1..].join(" ");
                    match self.client.load_model(&spec) {
                        Ok((alias, _)) => {
                            self.add_system_message(&format!("Loaded model: {}", alias));
                            self.refresh_models();
                        }
                        Err(e) => {
                            self.add_system_message(&format!("Failed to load: {}", e));
                        }
                    }
                } else {
                    self.add_system_message("Usage: :load <alias:path> or :load <path>");
                }
            }
            Some("unload") => {
                if let Some(alias) = parts.get(1) {
                    match self.client.unload_model(alias) {
                        Ok(()) => {
                            self.add_system_message(&format!("Unloaded: {}", alias));
                            self.refresh_models();
                        }
                        Err(e) => {
                            self.add_system_message(&format!("Failed: {}", e));
                        }
                    }
                }
            }
            Some("temp") | Some("temperature") => {
                if let Some(val) = parts.get(1).and_then(|s| s.parse::<f32>().ok()) {
                    self.temperature = val.clamp(0.0, 2.0);
                    self.status_message = format!("Temperature: {:.1}", self.temperature);
                } else {
                    self.add_system_message(&format!("Temperature: {:.1}", self.temperature));
                }
            }
            Some("tokens") | Some("max") => {
                if let Some(val) = parts.get(1).and_then(|s| s.parse::<u32>().ok()) {
                    self.max_tokens = val;
                    self.status_message = format!("Max tokens: {}", self.max_tokens);
                } else {
                    self.add_system_message(&format!("Max tokens: {}", self.max_tokens));
                }
            }
            Some("status") | Some("s") => {
                self.refresh_status();
                if let Some(ref status) = self.daemon_status {
                    self.add_system_message(&format!(
                        "Daemon v{} | Uptime: {}s | Models: {} | Requests: {}",
                        status.version,
                        status.uptime_secs,
                        status.models_loaded,
                        status.stats.requests_total
                    ));
                }
            }
            Some("help") | Some("h") | Some("?") => {
                self.show_help = true;
            }
            Some(cmd) => {
                self.add_system_message(&format!("Unknown command: {}", cmd));
            }
            None => {}
        }

        self.cursor_pos = 0;
    }

    fn add_system_message(&mut self, content: &str) {
        self.messages.push(Message {
            role: Role::System,
            content: content.to_string(),
            model: None,
            tokens: None,
            duration_ms: None,
        });
    }

    fn refresh_status(&mut self) {
        if let Ok(status) = self.client.status() {
            self.daemon_status = Some(status);
        }
    }

    fn refresh_models(&mut self) {
        if let Ok(models) = self.client.list_models() {
            self.models = models;
            // Set selected model to default if not set
            if self.selected_model.is_none() {
                self.selected_model = self
                    .models
                    .iter()
                    .find(|m| m.is_default)
                    .map(|m| m.alias.clone());
            }
        }
    }

    fn draw(&mut self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(10),   // Chat
                Constraint::Length(3), // Input
                Constraint::Length(1), // Status
            ])
            .split(f.area());

        self.draw_header(f, chunks[0]);
        self.draw_chat(f, chunks[1]);
        self.draw_input(f, chunks[2]);
        self.draw_status(f, chunks[3]);

        if self.show_help {
            self.draw_help_popup(f);
        }

        if self.input_mode == InputMode::ModelSelect {
            self.draw_model_popup(f);
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let model_name = self
            .selected_model
            .as_deref()
            .or_else(|| {
                self.models
                    .iter()
                    .find(|m| m.is_default)
                    .map(|m| m.alias.as_str())
            })
            .unwrap_or("none");

        let models_count = self.models.len();

        let header = Paragraph::new(Line::from(vec![
            Span::styled(
                " Mullama ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("│ Model: "),
            Span::styled(
                model_name,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" │ "),
            Span::styled(
                format!("{} models", models_count),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" │ "),
            Span::styled(
                format!("temp={:.1}", self.temperature),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" │ "),
            Span::styled(
                format!("max={}", self.max_tokens),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

        f.render_widget(header, area);
    }

    fn draw_chat(&mut self, f: &mut Frame, area: Rect) {
        let inner_height = area.height.saturating_sub(2) as usize;

        let items: Vec<ListItem> = self
            .messages
            .iter()
            .flat_map(|msg| {
                let (prefix, style, content_style) = match msg.role {
                    Role::User => (
                        "You",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                        Style::default().fg(Color::White),
                    ),
                    Role::Assistant => (
                        msg.model.as_deref().unwrap_or("AI"),
                        Style::default()
                            .fg(Color::Blue)
                            .add_modifier(Modifier::BOLD),
                        Style::default().fg(Color::White),
                    ),
                    Role::System => (
                        "System",
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::ITALIC),
                        Style::default().fg(Color::Yellow),
                    ),
                };

                let mut lines = vec![ListItem::new(Line::from(vec![
                    Span::styled(format!("┌─ {} ", prefix), style),
                    if let (Some(tokens), Some(ms)) = (msg.tokens, msg.duration_ms) {
                        Span::styled(
                            format!("[{} tokens, {}ms]", tokens, ms),
                            Style::default().fg(Color::DarkGray),
                        )
                    } else {
                        Span::raw("")
                    },
                ]))];

                for line in msg.content.lines() {
                    lines.push(ListItem::new(Line::from(vec![
                        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                        Span::styled(line, content_style),
                    ])));
                }

                lines.push(ListItem::new(Line::from(Span::styled(
                    "└─",
                    Style::default().fg(Color::DarkGray),
                ))));
                lines.push(ListItem::new(Line::from("")));

                lines
            })
            .collect();

        let total_items = items.len();
        self.messages_scroll = self
            .messages_scroll
            .min(total_items.saturating_sub(inner_height));

        self.messages_scroll_state = self
            .messages_scroll_state
            .content_length(total_items)
            .position(self.messages_scroll);

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(" Chat "),
            )
            .scroll((self.messages_scroll as u16, 0));

        f.render_widget(list, area);

        // Scrollbar
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None),
            area.inner(ratatui::layout::Margin {
                vertical: 1,
                horizontal: 0,
            }),
            &mut self.messages_scroll_state,
        );
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let (title, border_color) = match self.input_mode {
            InputMode::Insert => (" Message (Enter to send) ", Color::Green),
            InputMode::Normal => (" NORMAL (i to insert) ", Color::Yellow),
            InputMode::Command => (" Command ", Color::Magenta),
            InputMode::ModelSelect => (" Select Model ", Color::Cyan),
        };

        let display_input = if self.input_mode == InputMode::Command {
            format!(":{}", self.input)
        } else {
            self.input.clone()
        };

        let input = Paragraph::new(display_input.as_str())
            .style(Style::default().fg(Color::White))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(title),
            );

        f.render_widget(input, area);

        // Cursor
        if self.input_mode == InputMode::Insert || self.input_mode == InputMode::Command {
            let cursor_offset = if self.input_mode == InputMode::Command {
                1
            } else {
                0
            };
            f.set_cursor_position((
                area.x + self.cursor_pos as u16 + 1 + cursor_offset,
                area.y + 1,
            ));
        }
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let status = Paragraph::new(Line::from(vec![
            Span::raw(" "),
            if self.generating {
                Span::styled("● Generating", Style::default().fg(Color::Yellow))
            } else {
                Span::styled("●", Style::default().fg(Color::Green))
            },
            Span::raw(" │ "),
            Span::styled(&self.status_message, Style::default().fg(Color::DarkGray)),
            Span::raw(" │ "),
            Span::styled(
                "?: help  Ctrl+M: models  Ctrl+L: clear  Ctrl+Q: quit",
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .style(Style::default().bg(Color::Rgb(30, 30, 30)));

        f.render_widget(status, area);
    }

    fn draw_help_popup(&self, f: &mut Frame) {
        let area = centered_rect(60, 70, f.area());
        f.render_widget(Clear, area);

        let help_text = vec![
            Line::from(Span::styled(
                "Mullama TUI Help",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Keyboard Shortcuts:",
                Style::default().add_modifier(Modifier::UNDERLINED),
            )),
            Line::from("  Ctrl+Q       Quit"),
            Line::from("  Ctrl+C       Cancel generation / Quit"),
            Line::from("  Ctrl+L       Clear chat"),
            Line::from("  Ctrl+M       Select model"),
            Line::from("  ?            Toggle help"),
            Line::from("  Esc          Normal mode"),
            Line::from("  i/a          Insert mode"),
            Line::from("  :            Command mode"),
            Line::from(""),
            Line::from(Span::styled(
                "Commands:",
                Style::default().add_modifier(Modifier::UNDERLINED),
            )),
            Line::from("  :model <n>   Select model"),
            Line::from("  :models      List models"),
            Line::from("  :load <p>    Load model (alias:path)"),
            Line::from("  :unload <n>  Unload model"),
            Line::from("  :temp <v>    Set temperature"),
            Line::from("  :tokens <n>  Set max tokens"),
            Line::from("  :status      Show daemon status"),
            Line::from("  :clear       Clear chat"),
            Line::from("  :quit        Quit"),
            Line::from(""),
            Line::from(Span::styled(
                "Press any key to close",
                Style::default().fg(Color::DarkGray),
            )),
        ];

        let help = Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title(" Help "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(help, area);
    }

    fn draw_model_popup(&self, f: &mut Frame) {
        let area = centered_rect(50, 50, f.area());
        f.render_widget(Clear, area);

        let items: Vec<ListItem> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let style = if i == self.model_select_index {
                    Style::default().bg(Color::Blue).fg(Color::White)
                } else {
                    Style::default()
                };

                let marker = if model.is_default { " *" } else { "" };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {}{} ", model.alias, marker), style),
                    Span::styled(
                        format!("({:.0}B params)", model.info.parameters as f64 / 1e9),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Select Model (Enter to confirm, Esc to cancel) "),
        );

        f.render_widget(list, area);
    }
}

/// Helper to create a centered rect
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
