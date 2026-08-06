"""Stylesheet generation.

The old assets/style.qss covered seven widget classes, which is why every
dialog fell back to default Qt chrome and looked unrelated to the main
window. This generates the sheet from theme tokens and covers the classes
the app actually instantiates, so a new dialog matches without any
per-dialog styling.

`asset_urls` supplies file URLs for sub-control indicators (combo arrows,
checkbox marks). It is optional: when a key is absent the corresponding
rule is omitted and Qt draws its own indicator.
"""

from __future__ import annotations

from ui.theme.tokens import Theme, css, fade, hexs, mix, with_alpha


def _indicator_rules(theme: Theme, urls: dict[str, str]) -> str:
    """Image rules for sub-controls, emitted only for supplied assets."""
    out: list[str] = []

    down = urls.get("chevron_down")
    if down:
        out.append(
            f"""
QComboBox::down-arrow,
QDateEdit::down-arrow,
QTimeEdit::down-arrow {{
    image: url({down});
    width: 12px;
    height: 12px;
}}
QComboBox::down-arrow:disabled {{
    image: url({urls.get("chevron_down_dim", down)});
}}
"""
        )

    up_sm = urls.get("caret_up")
    down_sm = urls.get("caret_down")
    if up_sm and down_sm:
        out.append(
            f"""
QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {{
    image: url({up_sm});
    width: 9px;
    height: 9px;
}}
QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {{
    image: url({down_sm});
    width: 9px;
    height: 9px;
}}
"""
        )

    check = urls.get("check")
    if check:
        out.append(
            f"""
QCheckBox::indicator:checked,
QMenu::indicator:checked,
QListWidget::indicator:checked {{
    image: url({check});
}}
"""
        )

    return "".join(out)


def build_qss(theme: Theme, asset_urls: dict[str, str] | None = None) -> str:
    c = theme.palette
    m = theme.metrics
    t = theme.type
    urls = asset_urls or {}

    # Derived shades used in several places.
    field_bg = css(c.surface_alt)
    field_bg_focus = css(mix(c.surface_alt, c.accent, 0.10))
    disabled_text = css(c.text_faint)
    hover = css(c.surface_hi)
    # Neutral, and derived from `text` so it inverts with the theme for free:
    # near-white over dark, near-black over light. This is the generic pressed
    # state for quiet controls; the rules that genuinely want an accent press
    # (primary, send, toggles) set it themselves rather than making everything
    # else inherit a tint.
    pressed = css(with_alpha(c.text, 34))
    sel_bg = css(with_alpha(c.accent, 70))
    track = css(with_alpha(c.text_faint, 60))

    return f"""
/* ---------------------------------------------------------------- base */

* {{
    outline: 0;
}}

QWidget {{
    background: transparent;
    color: {css(c.text)};
    font-family: "{t.body}";
    font-size: {t.size_md}pt;
}}

QWidget:disabled {{
    color: {disabled_text};
}}

/* The main window paints its own backdrop in RoundedContainer, at whatever
   alpha the native chrome says is safe. An opaque background here would sit
   under everything and hide the acrylic blur entirely -- this rule is the
   single most likely reason for "every ctypes call succeeded and the window
   looks exactly the same". Dialogs are never frameless and stay opaque. */
QMainWindow {{
    background: transparent;
}}

QDialog {{
    background: {css(c.bg)};
    font-size: {t.size_md}pt;
}}

QToolTip {{
    background: {css(c.surface_alt, 250)};
    color: {css(c.text)};
    border: {m.border_w}px solid {css(c.border_strong)};
    border-radius: {m.radius_sm}px;
    padding: 5px 8px;
    font-family: "{t.body}";
    font-size: {t.size_sm}pt;
}}

/* ------------------------------------------------------------- surfaces */

QFrame#panel, QWidget#panel {{
    background: {css(c.surface)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_lg}px;
}}

QFrame#panelFlat, QWidget#panelFlat {{
    background: {css(c.surface_alt)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
}}

QFrame#divider {{
    background: {css(c.border)};
    border: none;
    max-height: 1px;
    min-height: 1px;
}}

QFrame#dividerV {{
    background: {css(c.border)};
    border: none;
    max-width: 1px;
    min-width: 1px;
}}

/* ---------------------------------------------------------------- text */

QLabel {{
    background: transparent;
    color: {css(c.text)};
}}

QLabel#heading {{
    font-family: "{t.display}";
    font-size: {t.size_lg}pt;
    color: {css(c.text)};
}}

QLabel#sectionTitle {{
    font-family: "{t.display}";
    font-size: {t.size_sm}pt;
    font-weight: 600;
    color: {css(c.text_dim)};
}}

QLabel#caption, QLabel#hint {{
    color: {css(c.text_dim)};
    font-size: {t.size_sm}pt;
}}

QLabel#readout {{
    font-family: "{t.mono}";
    font-size: {t.size_sm}pt;
    color: {css(c.text_dim)};
}}

QLabel#accentText {{
    color: {css(c.accent)};
}}

QLabel:disabled {{
    color: {disabled_text};
}}

/* -------------------------------------------------------------- inputs */

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {field_bg};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
    padding: 7px 10px;
    color: {css(c.text)};
    selection-background-color: {sel_bg};
    selection-color: {css(c.text)};
}}

QPlainTextEdit, QTextEdit {{
    font-family: "{t.mono}";
    font-size: {t.size_sm}pt;
}}

QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover {{
    border: {m.border_w}px solid {css(c.border_strong)};
}}

QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    background: {field_bg_focus};
    border: {m.border_w}px solid {css(c.accent)};
}}

QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {css(c.surface_alt, 120)};
    border: {m.border_w}px solid {css(c.border, 24)};
    color: {disabled_text};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {css(with_alpha(c.accent, 20))};
    border: none;
    border-left: {m.border_w}px solid {css(c.border)};
    width: 18px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    border-top-right-radius: {m.radius_md}px;
    border-bottom: {m.border_w}px solid {css(c.border)};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    border-bottom-right-radius: {m.radius_md}px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {hover};
}}

/* ------------------------------------------------------------- buttons */

QPushButton {{
    background: {css(with_alpha(c.accent, 24))};
    border: {m.border_w}px solid {css(c.border_strong)};
    border-radius: {m.radius_md}px;
    padding: 7px 14px;
    color: {css(c.text)};
    min-height: {m.control_h - 16}px;
}}

QPushButton:hover {{
    background: {hover};
    border: {m.border_w}px solid {css(c.accent)};
}}

QPushButton:pressed {{
    background: {pressed};
}}

QPushButton:checked {{
    background: {css(with_alpha(c.accent, 64))};
    border: {m.border_w}px solid {css(c.accent)};
    color: {css(c.accent_hot)};
}}

QPushButton:disabled {{
    background: {css(with_alpha(c.text_faint, 18))};
    border: {m.border_w}px solid {css(c.border, 22)};
    color: {disabled_text};
}}

QPushButton#primary {{
    background: {css(with_alpha(c.accent, 92))};
    border: {m.border_w}px solid {css(c.accent)};
    color: {css(c.accent_hot) if c.is_dark else hexs(c.surface)};
}}

QPushButton#primary:hover {{
    background: {css(with_alpha(c.accent, 128))};
}}

QPushButton#ghost {{
    background: transparent;
    border: {m.border_w}px solid {css(c.border)};
}}

QPushButton#ghost:hover {{
    background: {hover};
}}

QPushButton#danger {{
    background: {css(with_alpha(c.danger, 34))};
    border: {m.border_w}px solid {css(with_alpha(c.danger, 150))};
}}

QPushButton#danger:hover {{
    background: {css(with_alpha(c.danger, 70))};
}}

QToolButton {{
    background: {css(with_alpha(c.accent, 18))};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
    padding: 5px;
}}

QToolButton:hover {{
    background: {hover};
    border: {m.border_w}px solid {css(c.border_strong)};
}}

QToolButton:pressed {{
    background: {pressed};
}}

QToolButton:checked {{
    background: {css(with_alpha(c.accent, 64))};
    border: {m.border_w}px solid {css(c.accent)};
}}

QToolButton:disabled {{
    background: {css(with_alpha(c.text_faint, 14))};
    border: {m.border_w}px solid {css(c.border, 22)};
}}

QToolButton#flat {{
    background: transparent;
    border: none;
    border-radius: {m.radius_sm}px;
}}

QToolButton#flat:hover {{
    background: {hover};
}}

/* Window buttons follow the OS convention rather than the app's: square, no
   radius, no gaps, so the close button occupies the true window corner. */
QToolButton#windowButton, QToolButton#windowClose {{
    background: transparent;
    border: none;
    border-radius: 0;
}}

QToolButton#windowButton:hover {{
    background: {hover};
}}

QToolButton#windowButton:pressed {{
    background: {pressed};
}}

QToolButton#windowClose:hover {{
    background: {css(with_alpha(c.danger, 200))};
}}

QToolButton#windowClose:pressed {{
    background: {css(with_alpha(c.danger, 140))};
}}

QToolButton::menu-indicator {{
    image: none;
    width: 0;
}}

/* ------------------------------------------------------------ combobox */

QComboBox {{
    background: {field_bg};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
    padding: 6px 10px;
    color: {css(c.text)};
    min-height: {m.control_h - 16}px;
}}

QComboBox:hover {{
    border: {m.border_w}px solid {css(c.border_strong)};
}}

QComboBox:focus, QComboBox:on {{
    border: {m.border_w}px solid {css(c.accent)};
}}

QComboBox:disabled {{
    color: {disabled_text};
    background: {css(c.surface_alt, 120)};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: none;
    border-left: {m.border_w}px solid {css(c.border)};
}}

QComboBox QAbstractItemView {{
    background: {css(c.surface_alt, 252)};
    border: {m.border_w}px solid {css(c.border_strong)};
    border-radius: {m.radius_md}px;
    padding: 4px;
    color: {css(c.text)};
    selection-background-color: {sel_bg};
    selection-color: {css(c.text)};
    outline: 0;
}}

/* ---------------------------------------------------------------- menu */

QMenu {{
    background: {css(c.surface_alt, 252)};
    border: {m.border_w}px solid {css(c.border_strong)};
    border-radius: {m.radius_md}px;
    padding: 6px;
    color: {css(c.text)};
}}

QMenu::item {{
    background: transparent;
    padding: 7px 26px 7px 12px;
    border-radius: {m.radius_sm}px;
}}

QMenu::item:selected {{
    background: {hover};
    color: {css(c.accent_hot) if c.is_dark else css(c.text)};
}}

QMenu::item:disabled {{
    color: {disabled_text};
}}

QMenu::separator {{
    height: 1px;
    background: {css(c.border)};
    margin: 5px 8px;
}}

QMenu::icon {{
    padding-left: 8px;
}}

/* --------------------------------------------------------------- lists */

QListWidget, QListView, QTreeView, QTableView {{
    background: {css(c.surface_alt)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
    padding: 4px;
    color: {css(c.text)};
    outline: 0;
    alternate-background-color: {css(c.surface)};
}}

QListWidget::item, QListView::item, QTreeView::item {{
    padding: 6px 8px;
    border-radius: {m.radius_sm}px;
}}

QListWidget::item:hover, QListView::item:hover, QTreeView::item:hover {{
    background: {hover};
}}

QListWidget::item:selected, QListView::item:selected, QTreeView::item:selected {{
    background: {sel_bg};
    color: {css(c.text)};
}}

QHeaderView::section {{
    background: {css(c.surface)};
    border: none;
    border-bottom: {m.border_w}px solid {css(c.border)};
    padding: 6px 8px;
    color: {css(c.text_dim)};
    font-family: "{t.mono}";
}}

/* ----------------------------------------------------------- indicators */

QProgressBar {{
    background: {track};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: {css(c.text_dim)};
}}

QProgressBar::chunk {{
    background: {css(c.accent)};
    border-radius: 3px;
}}

QProgressBar#subtle::chunk {{
    background: {css(c.accent_dim)};
}}

QCheckBox, QRadioButton {{
    background: transparent;
    color: {css(c.text)};
    spacing: 9px;
    padding: 3px 0;
}}

QCheckBox:disabled, QRadioButton:disabled {{
    color: {disabled_text};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 17px;
    height: 17px;
    background: {field_bg};
    border: {m.border_w}px solid {css(c.border_strong)};
}}

QCheckBox::indicator {{
    border-radius: {m.radius_sm - 2}px;
}}

QRadioButton::indicator {{
    border-radius: 9px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border: {m.border_w}px solid {css(c.accent)};
    background: {hover};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {css(c.accent)};
    border: {m.border_w}px solid {css(c.accent)};
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: {css(with_alpha(c.text_faint, 26))};
    border: {m.border_w}px solid {css(c.border, 26)};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {track};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {css(c.accent)};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    background: {css(c.accent_hot)};
    border: {m.border_w}px solid {css(c.accent)};
}}

QSlider::handle:horizontal:hover {{
    background: {css(c.accent)};
}}

/* ------------------------------------------------------------- scrolls */

QScrollArea, QAbstractScrollArea {{
    background: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px 2px 2px 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0 2px 2px 2px;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {css(with_alpha(c.text_faint, 110))};
    border-radius: 4px;
    min-height: 28px;
    min-width: 28px;
}}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {css(with_alpha(c.accent, 150))};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
    background: none;
    border: none;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* --------------------------------------------------------------- boxes */

QGroupBox {{
    background: {css(c.surface_alt, 140)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
    margin-top: 16px;
    padding: 12px 10px 10px 10px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: {css(c.text_dim)};
    font-family: "{t.display}";
    font-size: {t.size_sm}pt;
}}

QTabWidget::pane {{
    background: {css(c.surface)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
    top: -1px;
}}

QTabBar::tab {{
    background: transparent;
    border: {m.border_w}px solid transparent;
    border-radius: {m.radius_sm}px;
    padding: 7px 14px;
    margin-right: 4px;
    color: {css(c.text_dim)};
}}

QTabBar::tab:hover {{
    background: {hover};
    color: {css(c.text)};
}}

QTabBar::tab:selected {{
    background: {css(with_alpha(c.accent, 46))};
    border: {m.border_w}px solid {css(c.border_strong)};
    color: {css(c.text)};
}}

QSplitter::handle {{
    background: {css(c.border)};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

/* ------------------------------------------------------------ app bar */

QWidget#appBar {{
    background: transparent;
    border-bottom: {m.border_w}px solid {css(c.border, 30)};
}}

QLabel#brand {{
    color: {css(c.text)};
}}

QToolButton#barButton {{
    background: transparent;
    border: none;
    border-radius: {m.radius_sm}px;
}}

QToolButton#barButton:hover {{
    background: {hover};
}}

QToolButton#barButton:checked {{
    background: {css(with_alpha(c.accent, 46))};
}}

/* ------------------------------------------------------------ orb area */

QLabel#greeting {{
    color: {css(c.text)};
    background: transparent;
}}

QLabel#statePill {{
    color: {css(c.text_faint)};
    background: transparent;
}}

/* ------------------------------------------------------ conversation */

QWidget#chatContainer {{
    background: transparent;
}}

/* Neutral, not accent-tinted. The user's turn is the one surface that
   repeats down the whole transcript, so tinting it painted the entire
   conversation gold. It lifts by luminance instead. */
QFrame#userTurn {{
    background: {css(c.surface_alt)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_lg}px;
}}

QFrame#assistantTurn {{
    background: transparent;
    border: none;
}}

QLabel#turnRole {{
    color: {css(c.text_dim)};
    background: transparent;
}}

QLabel#turnText {{
    color: {css(c.text)};
    background: transparent;
    line-height: 150%;
}}

QLabel#turnIntent {{
    color: {css(c.text_faint)};
    background: transparent;
}}

QLabel#turnImage {{
    background: transparent;
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
    padding: 3px;
}}

/* --------------------------------------------------------- composer */

QFrame#composer {{
    background: {css(c.surface_alt)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_xl}px;
}}

/* The composer is where the accent belongs: it is the one focused element
   on the screen, and spending the accent here is what lets every other
   border stay neutral. */
QFrame#composer:focus-within {{
    border: {m.border_w}px solid {css(with_alpha(c.accent, 150))};
}}

QLineEdit#composerInput {{
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 6px 8px;
    color: {css(c.text)};
    selection-background-color: {sel_bg};
}}

QLineEdit#composerInput:focus {{
    background: transparent;
    border: none;
}}

QLineEdit#composerInput:disabled {{
    background: transparent;
    border: none;
    color: {disabled_text};
}}

QToolButton#composerButton, QPushButton#composerButton {{
    background: transparent;
    border: none;
    border-radius: {m.radius_md}px;
}}

QToolButton#composerButton:hover, QPushButton#composerButton:hover {{
    background: {hover};
}}

QToolButton#composerButton:disabled, QPushButton#composerButton:disabled {{
    background: transparent;
}}

QToolButton#composerToggle {{
    background: transparent;
    border: none;
    border-radius: {m.radius_md}px;
}}

QToolButton#composerToggle:hover {{
    background: {hover};
}}

QToolButton#composerToggle:checked {{
    background: {css(with_alpha(c.accent, 58))};
}}

QPushButton#composerSend {{
    background: {css(with_alpha(c.accent, 40))};
    border: none;
    border-radius: {m.radius_md}px;
}}

QPushButton#composerSend:hover {{
    background: {css(with_alpha(c.accent, 92))};
}}

QPushButton#composerSend:disabled {{
    background: {css(with_alpha(c.text_faint, 20))};
}}

QFrame#attachmentChip {{
    background: {css(with_alpha(c.accent, 20))};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_md}px;
}}

QLabel#turnImage, QFrame#attachmentChip QLabel {{
    background: transparent;
}}

/* ------------------------------------------------------- diagnostics */

QFrame#diagnosticsPanel {{
    background: {css(c.surface)};
    border: none;
    border-left: {m.border_w}px solid {css(c.border, 40)};
}}

/* ------------------------------------------------------------- dialogs */

QMessageBox {{
    background: {css(c.bg)};
}}

QMessageBox QLabel {{
    color: {css(c.text)};
}}

QDialogButtonBox QPushButton {{
    min-width: 88px;
}}

/* ------------------------------------------------------------ settings */

QListWidget#settingsRail {{
    background: {css(c.surface)};
    border: none;
    padding: 10px 8px;
    outline: 0;
}}

QListWidget#settingsRail::item {{
    padding: 8px 12px;
    border-radius: {m.radius_md}px;
    color: {css(c.text_dim)};
}}

QListWidget#settingsRail::item:hover {{
    background: {hover};
    color: {css(c.text)};
}}

/* The selected page is the one place on this window the accent is spent. */
QListWidget#settingsRail::item:selected {{
    background: {css(with_alpha(c.accent, 46))};
    color: {css(c.text)};
}}

QLabel#pageTitle {{
    color: {css(c.text)};
    background: transparent;
}}

QFrame#settingsCard {{
    background: {css(c.surface)};
    border: {m.border_w}px solid {css(c.border)};
    border-radius: {m.radius_lg}px;
}}

QFrame#settingRow {{
    background: transparent;
    border: none;
}}

QFrame#rowDivider {{
    background: {css(c.border)};
    border: none;
}}

QLabel#rowTitle {{
    color: {css(c.text)};
    background: transparent;
}}

QLabel#rowDescription {{
    color: {css(c.text_dim)};
    background: transparent;
}}

QWidget#settingsFooter {{
    background: {css(c.surface)};
    border-top: {m.border_w}px solid {css(c.border)};
}}

QFrame#banner {{
    background: {css(with_alpha(c.warn, 34))};
    border: {m.border_w}px solid {css(with_alpha(c.warn, 90))};
    border-radius: {m.radius_md}px;
}}

QFrame#banner[tone="error"] {{
    background: {css(with_alpha(c.danger, 34))};
    border: {m.border_w}px solid {css(with_alpha(c.danger, 90))};
}}

QFrame#banner QLabel {{
    background: transparent;
    color: {css(c.text)};
}}

QFrame#accentSwatch {{
    border: {m.border_w}px solid {css(c.border_strong)};
}}

{_indicator_rules(theme, urls)}
""".strip()
