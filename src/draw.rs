use crate::game::{Game, Dir};
use crate::pos::Pos;
use ratatui::text::{Spans, Span};
use ratatui::style::{Color, Style, Modifier};

/// Render the current game state as styled Spans with colors
pub fn render_game(game: &Game) -> Vec<Spans<'static>> {
    let mut lines = Vec::new();

    // Title/Header with ASCII art
    if game.alive {
        if game.pause {
            lines.push(Spans::from(vec![
                Span::styled("╔══════════════════════════════════════════╗", Style::default().fg(Color::Yellow)),
            ]));
            lines.push(Spans::from(vec![
                Span::styled("║          🐍 SNAKE GAME - PAUSED ⏸        ║", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ]));
            lines.push(Spans::from(vec![
                Span::styled("╚══════════════════════════════════════════╝", Style::default().fg(Color::Yellow)),
            ]));
        } else {
            lines.push(Spans::from(vec![
                Span::styled("╔══════════════════════════════════════════╗", Style::default().fg(Color::Cyan)),
            ]));
            lines.push(Spans::from(vec![
                Span::styled("║            🐍 SNAKE GAME 🍎              ║", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            ]));
            lines.push(Spans::from(vec![
                Span::styled("╚══════════════════════════════════════════╝", Style::default().fg(Color::Cyan)),
            ]));
        }
    } else {
        lines.push(Spans::from(vec![
            Span::styled("╔══════════════════════════════════════════╗", Style::default().fg(Color::Red)),
        ]));
        lines.push(Spans::from(vec![
            Span::styled("║          💀 GAME OVER! 💀                ║", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        ]));
        lines.push(Spans::from(vec![
            Span::styled("╚══════════════════════════════════════════╝", Style::default().fg(Color::Red)),
        ]));
    }

    lines.push(Spans::from(""));

    // Top border with decorative corners
    let mut top_border = vec![Span::styled("╔", Style::default().fg(Color::White))];
    for _ in 0..game.width - 2 {
        top_border.push(Span::styled("═", Style::default().fg(Color::White)));
    }
    top_border.push(Span::styled("╗", Style::default().fg(Color::White)));
    lines.push(Spans::from(top_border));

    // Middle - game field
    for y in 1..game.height - 1 {
        let mut row = vec![Span::styled("║", Style::default().fg(Color::White))];
        
        for x in 1..game.width - 1 {
            let p = Pos::new(x, y);
            
            // Draw grid pattern
            let is_grid_line = (x % 2 == 0) || (y % 2 == 0);
            
            if p == game.apple {
                // Red apple
                row.push(Span::styled("●", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)));
            } else if game.snake_contains(p) {
                let head = game.snake.front().unwrap();
                if *head == p {
                    // Snake head - different look based on direction
                    let head_char = match game.dir {
                        Dir::Up => "▲",
                        Dir::Down => "▼",
                        Dir::Left => "◄",
                        Dir::Right => "►",
                    };
                    row.push(Span::styled(head_char, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)));
                } else {
                    // Snake body
                    row.push(Span::styled("●", Style::default().fg(Color::Rgb(50, 205, 50))));
                }
            } else {
                // Empty space with grid pattern
                if is_grid_line {
                    row.push(Span::styled("·", Style::default().fg(Color::Rgb(40, 40, 40))));
                } else {
                    if (x + y) % 4 == 0 {
                        row.push(Span::styled(" ", Style::default().bg(Color::Rgb(10, 10, 10))));
                    } else {
                        row.push(Span::styled(" ", Style::default().bg(Color::Rgb(5, 5, 5))));
                    }
                }
            }
        }
        
        row.push(Span::styled("║", Style::default().fg(Color::White)));
        lines.push(Spans::from(row));
    }

    // Bottom border with decorative corners
    let mut bottom_border = vec![Span::styled("╚", Style::default().fg(Color::White))];
    for _ in 0..game.width - 2 {
        bottom_border.push(Span::styled("═", Style::default().fg(Color::White)));
    }
    bottom_border.push(Span::styled("╝", Style::default().fg(Color::White)));
    lines.push(Spans::from(bottom_border));

    lines.push(Spans::from(""));

    // Score and info with colors and icons
    lines.push(Spans::from(vec![
        Span::styled("📊 Score: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{}", game.score), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled("  │  ", Style::default().fg(Color::DarkGray)),
        Span::styled("Length: ", Style::default().fg(Color::Cyan)),
        Span::styled(format!("{}", game.snake.len()), Style::default().fg(Color::Green)),
    ]));

    lines.push(Spans::from(""));

    // Controls with decorative icons
    if game.alive {
        lines.push(Spans::from(vec![
            Span::styled("⌨  Controls: ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        ]));
        lines.push(Spans::from(vec![
            Span::styled("  ↑/W ", Style::default().fg(Color::Green)),
            Span::styled("Up  ", Style::default().fg(Color::Gray)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled("↓/S ", Style::default().fg(Color::Green)),
            Span::styled("Down  ", Style::default().fg(Color::Gray)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled("←/A ", Style::default().fg(Color::Green)),
            Span::styled("Left  ", Style::default().fg(Color::Gray)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled("→/D ", Style::default().fg(Color::Green)),
            Span::styled("Right", Style::default().fg(Color::Gray)),
        ]));
        lines.push(Spans::from(vec![
            Span::styled("  P ", Style::default().fg(Color::Yellow)),
            Span::styled("Pause  ", Style::default().fg(Color::Gray)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
            Span::styled("Q ", Style::default().fg(Color::Red)),
            Span::styled("Quit", Style::default().fg(Color::Gray)),
        ]));
    } else {
        lines.push(Spans::from(vec![
            Span::styled("⌨  Press ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::styled("R ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled("to restart or ", Style::default().fg(Color::White)),
            Span::styled("Q ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled("to quit", Style::default().fg(Color::White)),
        ]));
    }

    lines
}
