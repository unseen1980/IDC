#!/usr/bin/env python3
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from PySide6.QtCharts import (
    QChart, QChartView, QBarSeries, QBarSet, QValueAxis, QBarCategoryAxis
)
from PySide6.QtCore import QPointF, QMargins
from PySide6.QtGui import QFont, QPainter


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_squad2_e2e.sh"


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata describing a dataset entry exposed in the UI."""

    label: str
    script: Path
    dataset_type: str  # e.g. 'corpus', 'idc-single'
    default_path: Optional[Path] = None
    allow_browse: bool = True
    limit_applicable: bool = False
    enable_idc_params: bool = False
    doc_name_visible: bool = False
    path_env: Optional[str] = None
    stats_subdir: Optional[str] = None
    description: str = ""


class ChartPanel(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.chart = QChart()
        self.chart.setTitle(title)
        
        # Enable anti-aliasing for better text rendering
        self.view = QChartView(self.chart)
        self.view.setRenderHint(QPainter.Antialiasing)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        self.setLayout(layout)

    def render_bars(self, categories: List[str], series_data: Dict[str, float], y_max: float = 1.0):
        self.chart.removeAllSeries()
        
        if not categories or not series_data:
            return
        
        # Create bar series with separate bar set for each variant (for different colors)
        series = QBarSeries()
        
        # Track actual values for auto-scaling
        actual_values = []
        
        # Define colors for each variant
        from PySide6.QtGui import QColor
        colors = [
            "#3498db",  # Blue
            "#e74c3c",  # Red  
            "#2ecc71",  # Green
            "#f39c12",  # Orange
            "#9b59b6"   # Purple
        ]
        
        # Create separate bar set for each category for individual coloring
        for i, cat in enumerate(categories):
            value = series_data.get(cat, 0.0)
            actual_values.append(value)
            
            # Create a bar set with just this one value
            barset = QBarSet(cat)
            
            # Add value to this position and zeros to all other positions
            for j in range(len(categories)):
                if j == i:
                    barset.append(value)
                else:
                    barset.append(0.0)  # Zero for other positions
            
            # Set unique color for this bar set
            color = QColor(colors[i % len(colors)])
            barset.setColor(color)
            border_color = color.darker(120)  # Darker border
            barset.setBorderColor(border_color)
            
            series.append(barset)
        self.chart.addSeries(series)
        
        # Enable value labels on the series with enhanced visibility
        series.setLabelsVisible(True)
        series.setLabelsPosition(QBarSeries.LabelsOutsideEnd)
        
        # Set label precision and angle for better readability
        series.setLabelsAngle(0)
        
        # Set label format for the series with proper precision
        if "Length" in self.title or "Token" in self.title:
            series.setLabelsPrecision(1)  # 1 decimal for length
        else:
            series.setLabelsPrecision(3)  # 3 decimals for percentages
        
        
        # Set up axes
        axisX = QBarCategoryAxis()
        axisX.append(categories)  # Use variants as categories
        axisY = QValueAxis()
        
        # Smart Y-axis scaling with extra space for labels
        if actual_values:
            max_val = max(actual_values)
            min_val = min(actual_values)
            
            # For small ranges, zoom in for better visibility
            if max_val < 0.3:  # Low values overall
                y_min = max(0, min_val - 0.02)
                y_max_range = max_val * 1.4 + 0.05  # Extra space for labels
            elif max_val - min_val < 0.1:  # Small differences
                y_min = max(0, min_val - 0.05)
                y_max_range = max_val + 0.1  # More space for labels
            else:  # Normal range
                y_min = 0.0
                y_max_range = min(y_max * 1.3, max_val * 1.3)  # Extra space for labels
                
            axisY.setRange(y_min, y_max_range)
        
        # Simplify Y-axis labels to avoid confusion with bar labels
        if "Length" in self.title or "Token" in self.title:
            axisY.setLabelFormat("%.1f")
            # Show fewer ticks for cleaner appearance
            axisY.setTickCount(6)
        else:
            axisY.setLabelFormat("%.2f")  # Reduced precision on axis
            axisY.setTickCount(6)
        
        self.chart.addAxis(axisX, Qt.AlignBottom)
        self.chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisX)
        series.attachAxis(axisY)
        
        # Hide legend for individual charts since X-axis labels show variant names
        self.chart.legend().setVisible(False)
        
        # Make chart more compact with better margins
        self.chart.setMargins(QMargins(5, 5, 5, 5))
        
        # Set title font
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        self.chart.setTitleFont(title_font)
        
        self.view.repaint()


class ComprehensiveChartsPanel(QWidget):
    """Shows all metrics in a grid layout for easy comparison"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.charts = {}
        
        layout = QVBoxLayout(self)
        
        # Create responsive grid layout for charts
        grid = QGridLayout()
        grid.setSpacing(10)  # Add spacing between charts
        
        # Create individual chart panels
        metrics = [
            ("R@1 (span)", "r1", 1.0),
            ("R@5 (span)", "r5", 1.0),
            ("MRR (span)", "mrr", 1.0),
            ("Coverage", "cov", 1.0),
            ("Completeness", "completeness", 1.0),  # NEW: Extended metric
            ("Diversity", "diversity", 1.0),  # NEW: Extended metric
            ("Redundancy", "redundancy", 1.0),  # NEW: Extended metric (lower is better)
            ("Efficiency", "efficiency", 1.0),  # NEW: Extended metric
            ("Avg Length", "len", 20.0),
            ("Avg Tokens", "tok", 400.0),
        ]
        
        for i, (title, key, y_max) in enumerate(metrics):
            chart_panel = ChartPanel(title)
            chart_panel.setMinimumHeight(250)  # Reduced minimum height for better fit
            chart_panel.setMaximumHeight(350)  # Reasonable maximum height
            chart_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.charts[key] = (chart_panel, y_max)
            
            # Arrange in 5x2 grid for all metrics including extended ones
            row, col = divmod(i, 2)  # 2 columns for better spacing
            grid.addWidget(chart_panel, row, col)
        
        # Make columns and rows stretch equally for responsive design
        for col in range(2):  # Now 2 columns
            grid.setColumnStretch(col, 1)
        rows = (len(metrics) + 1) // 2  # Calculate rows for 2-column layout
        for row in range(rows):
            grid.setRowStretch(row, 1)
        
        layout.addLayout(grid)
        
        # Add data table below charts with better sizing
        self.data_table = QTableWidget()
        self.data_table.setMinimumHeight(180)
        self.data_table.setMaximumHeight(300)  # Increased max height
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add label with better styling
        metrics_label = QLabel("📊 Detailed Metrics")
        metrics_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(metrics_label)
        layout.addWidget(self.data_table)
        
        self.setLayout(layout)
    
    def update_charts(self, variants: List[str], metrics_data: Dict[str, Dict[str, float]]):
        """Update all charts with new data"""
        if not variants or not metrics_data:
            # Clear all charts if no data provided
            self._clear_all_charts()
            return
            
        # Update individual metric charts
        for metric_key, (chart_panel, y_max) in self.charts.items():
            if metric_key == "summary":
                continue  # Handle summary separately
                
            data = metrics_data.get(metric_key, {})
            chart_panel.render_bars(variants, data, y_max)
        
        # Create summary chart with multiple metrics
        # self._create_summary_chart(variants, metrics_data)
        
        # Update data table
        self._update_data_table(variants, metrics_data)
    
    def _clear_all_charts(self):
        """Clear all charts and data table"""
        for _, (chart_panel, _) in self.charts.items():
            # Clear each chart
            chart_panel.chart.removeAllSeries()
            # Remove axes individually
            for axis in chart_panel.chart.axes():
                chart_panel.chart.removeAxis(axis)
            chart_panel.view.repaint()
        
        # Clear data table
        self.data_table.clear()
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
    
    def _update_data_table(self, variants: List[str], metrics_data: Dict[str, Dict[str, float]]):
        """Update the data table with precise numerical values"""
        metric_names = ["R@1", "R@5", "MRR", "Coverage", "Completeness", "Diversity", "Redundancy↓", "Efficiency", "Avg Length", "Avg Tokens"]
        metric_keys = ["r1", "r5", "mrr", "cov", "completeness", "diversity", "redundancy", "efficiency", "len", "tok"]
        
        self.data_table.setRowCount(len(variants))
        self.data_table.setColumnCount(len(metric_names))
        self.data_table.setHorizontalHeaderLabels(metric_names)
        self.data_table.setVerticalHeaderLabels(variants)
        
        # Populate table
        for row, variant in enumerate(variants):
            for col, metric_key in enumerate(metric_keys):
                value = metrics_data.get(metric_key, {}).get(variant, 0.0)
                if metric_key in {"len", "tok"}:  # Length/token show one decimal place
                    item_text = f"{value:.1f}"
                else:  # Other metrics show 3 decimal places
                    item_text = f"{value:.3f}"
                    
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                
                # Color coding for better visualization
                from PySide6.QtGui import QColor
                if metric_key in ["r1", "r5", "mrr", "cov"]:  # Higher is better
                    if value >= 0.7: 
                        item.setBackground(QColor(144, 238, 144))  # Light green
                    elif value >= 0.5: 
                        item.setBackground(QColor(255, 255, 224))  # Light yellow
                    elif value < 0.3: 
                        item.setBackground(QColor(211, 211, 211))  # Light gray
                elif metric_key == "len":  # Lower is better for average length
                    if value <= 4.0:  # Very good - short chunks
                        item.setBackground(QColor(144, 238, 144))  # Light green
                    elif value <= 6.0:  # Good - medium chunks
                        item.setBackground(QColor(255, 255, 224))  # Light yellow
                    elif value > 8.0:  # Not ideal - long chunks
                        item.setBackground(QColor(255, 182, 193))  # Light pink
                elif metric_key == "tok":
                    # Token counts scale with chunk size; keep similar thresholds but higher values.
                    if value <= 120.0:
                        item.setBackground(QColor(144, 238, 144))
                    elif value <= 200.0:
                        item.setBackground(QColor(255, 255, 224))
                    elif value > 260.0:
                        item.setBackground(QColor(255, 182, 193))
                
                self.data_table.setItem(row, col, item)
        
        # Make table columns responsive
        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # First column (variant names)
        for i in range(1, len(metric_names)):
            header.setSectionResizeMode(i, QHeaderView.Stretch)  # Other columns stretch
        
        # Enable better row sizing
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    # def _create_summary_chart(self, variants: List[str], metrics_data: Dict[str, Dict[str, float]]):
    #     """Create a multi-series chart showing key metrics together"""
    #     summary_chart, _ = self.charts["summary"]
    #     chart = summary_chart.chart
    #     chart.removeAllSeries()
    #     chart.setTitle("Key Metrics Comparison (All Variants)")
        
    #     # Create series for each key metric with enhanced visualization
    #     key_metrics = [("R@1", "r1"), ("MRR", "mrr"), ("Coverage", "cov")]
    #     colors = ["#3498db", "#e74c3c", "#2ecc71"]  # Blue, Red, Green
        
    #     series = QBarSeries()
    #     for i, (metric_name, metric_key) in enumerate(key_metrics):
    #         barset = QBarSet(metric_name)
    #         data = metrics_data.get(metric_key, {})
    #         for variant in variants:
    #             value = data.get(variant, 0.0)
    #             barset.append(value)
            
    #         # Set color
    #         if i < len(colors):
    #             barset.setColor(colors[i])
    #         series.append(barset)
        
    #     chart.addSeries(series)
        
    #     # Enable value labels on the series
    #     series.setLabelsVisible(True)
    #     series.setLabelsPosition(QBarSeries.LabelsOutsideEnd)
    #     series.setLabelsPrecision(3)  # 3 decimal places
        
    #     # Set smaller font for summary chart to fit more values
        
    #     # Set up axes with variant names
    #     axisX = QBarCategoryAxis()
    #     # Truncate long variant names for better display
    #     display_variants = [v[:12] + "..." if len(v) > 12 else v for v in variants]
    #     axisX.append(display_variants)
        
    #     axisY = QValueAxis()
    #     axisY.setRange(0.0, 1.1)  # Add padding
    #     axisY.setTitleText("Score")
        
    #     chart.addAxis(axisX, Qt.AlignBottom)
    #     chart.addAxis(axisY, Qt.AlignLeft)
    #     series.attachAxis(axisX)
    #     series.attachAxis(axisY)
    #     chart.legend().setVisible(True)
    #     chart.legend().setAlignment(Qt.AlignBottom)
        
    #     # Improve chart appearance
    #     chart.setMargins(QMargins(10, 10, 10, 10))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IDC Benchmark UI - Multi-Variant Comparison")
        
        # Make window responsive to screen size
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            # Use 85% of screen width and 80% of screen height
            width = int(screen_geometry.width() * 0.85)
            height = int(screen_geometry.height() * 0.80)
            # Set reasonable minimum and maximum bounds
            width = max(1200, min(width, 2000))
            height = max(800, min(height, 1400))
            self.resize(width, height)
        else:
            self.resize(1400, 1000)  # Fallback size

        self._last_auto_doc_name = ""
        # Controls
        self.dataset_combo = QComboBox()
        self.dataset_combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.dataset_combo.setMinimumContentsLength(0)
        self.dataset_combo.setMinimumWidth(0)
        self.dataset_combo.setInsertPolicy(QComboBox.NoInsert)
        self.dataset_combo.setMaxVisibleItems(24)
        self.dataset_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dataset_combo.setToolTip("Select an input dataset; hover entries to see full details.")
        combo_view = self.dataset_combo.view()
        if hasattr(combo_view, "setWordWrap"):
            combo_view.setWordWrap(True)

        self.refresh_datasets_btn = QPushButton("Refresh")
        self.refresh_datasets_btn.setToolTip("Rescan data folders for new entries")
        self.refresh_datasets_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.refresh_datasets_btn.clicked.connect(self._refresh_dataset_list)
        self.dataset_catalog = self._build_dataset_catalog()
        self.dataset_info_by_label = {info.label: info for info in self.dataset_catalog}
        self._refresh_dataset_items()
        self.dataset_combo.currentIndexChanged.connect(lambda _: self._on_dataset_changed())

        dataset_row = QHBoxLayout()
        dataset_row.setContentsMargins(0, 0, 0, 0)
        dataset_row.setSpacing(6)
        dataset_row.addWidget(self.dataset_combo, 1)
        dataset_row.addWidget(self.refresh_datasets_btn)
        self.dataset_combo_wrap = QWidget()
        self.dataset_combo_wrap.setLayout(dataset_row)

        self.doc_name_edit = QLineEdit()
        self.doc_name_edit.setPlaceholderText("Document name (e.g., 'Normans', 'Computational_complexity_theory')")
        self.doc_name_edit.setText("Normans")  # Default to Normans
        self.doc_name_edit.setVisible(False)  # Hidden by default

        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, 100000)
        self.limit_spin.setValue(int(os.environ.get("DEFAULT_DOC_LIMIT", "100")))
        self.limit_spin.setVisible(False)

        # Dataset file path (for Qasper)
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.textChanged.connect(self._on_dataset_path_changed)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_dataset)
        ds_h = QHBoxLayout(); ds_h.addWidget(self.dataset_path_edit, 1); ds_h.addWidget(self.browse_btn)
        self.ds_wrap = QWidget(); self.ds_wrap.setLayout(ds_h)

        # Variants
        self.chk_idc = QCheckBox("IDC")
        self.chk_fixed = QCheckBox("Fixed")
        self.chk_sliding = QCheckBox("Sliding")
        self.chk_coh = QCheckBox("Coherence")
        self.chk_para = QCheckBox("Paragraphs")
        for cb in [self.chk_idc, self.chk_fixed, self.chk_sliding, self.chk_coh, self.chk_para]:
            cb.setChecked(True)

        idc_group = QGroupBox("IDC Hyperparameters")
        idc_form = QFormLayout()
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setRange(0.00001, 1.0)
        self.lambda_spin.setSingleStep(0.01)
        self.lambda_spin.setValue(float(os.environ.get("LAMBDA", "0.1")))
        idc_form.addRow("Lambda", self.lambda_spin)

        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(4, 48)
        self.max_len_spin.setValue(int(os.environ.get("MAX_LEN", "20")))
        idc_form.addRow("Max sentences", self.max_len_spin)

        self.min_len_spin = QSpinBox()
        self.min_len_spin.setRange(1, 10)
        self.min_len_spin.setValue(int(os.environ.get("MIN_LEN", "3")))
        idc_form.addRow("Min sentences", self.min_len_spin)

        self.boundary_penalty_spin = QDoubleSpinBox()
        self.boundary_penalty_spin.setDecimals(2)
        self.boundary_penalty_spin.setRange(0.0, 2.0)
        self.boundary_penalty_spin.setSingleStep(0.05)
        self.boundary_penalty_spin.setValue(float(os.environ.get("BOUNDARY_PENALTY", "1.2")))
        idc_form.addRow("Boundary penalty", self.boundary_penalty_spin)

        self.coherence_weight_spin = QDoubleSpinBox()
        self.coherence_weight_spin.setDecimals(2)
        self.coherence_weight_spin.setRange(0.0, 1.0)
        self.coherence_weight_spin.setSingleStep(0.05)
        self.coherence_weight_spin.setValue(float(os.environ.get("COHERENCE_WEIGHT", "0.3")))
        idc_form.addRow("Coherence weight", self.coherence_weight_spin)

        self.auto_tune_chk = QCheckBox("Enable auto-tuning")
        self.auto_tune_chk.setChecked(os.environ.get("AUTO_TUNE", "0") == "1")
        idc_form.addRow(self.auto_tune_chk)

        # Advanced Optimizations
        adv_label = QLabel("<b>Advanced Optimizations</b>")
        idc_form.addRow(adv_label)

        self.contextual_emb_chk = QCheckBox("Contextual embeddings (recommended)")
        self.contextual_emb_chk.setChecked(os.environ.get("CONTEXTUAL_EMBEDDINGS", "1") == "1")
        self.contextual_emb_chk.setToolTip(
            "Include adjacent chunk context in embeddings for better retrieval.\n"
            "Improves R@5 by ~5% with minimal overhead."
        )
        idc_form.addRow(self.contextual_emb_chk)

        self.context_weight_spin = QDoubleSpinBox()
        self.context_weight_spin.setDecimals(2)
        self.context_weight_spin.setRange(0.0, 0.5)
        self.context_weight_spin.setSingleStep(0.05)
        self.context_weight_spin.setValue(float(os.environ.get("CONTEXT_WEIGHT", "0.15")))
        self.context_weight_spin.setToolTip(
            "Weight for contextual information from adjacent chunks.\n"
            "Short docs: 0.10 | Long docs: 0.20"
        )
        idc_form.addRow("  Context weight", self.context_weight_spin)

        self.density_aware_chk = QCheckBox("Density-aware segmentation (recommended)")
        self.density_aware_chk.setChecked(os.environ.get("DENSITY_AWARE", "1") == "1")
        self.density_aware_chk.setToolTip(
            "Allow longer chunks in dense content regions.\n"
            "Reduces token count by ~8-10% while maintaining coverage."
        )
        idc_form.addRow(self.density_aware_chk)

        self.density_discount_spin = QDoubleSpinBox()
        self.density_discount_spin.setDecimals(2)
        self.density_discount_spin.setRange(0.0, 1.0)
        self.density_discount_spin.setSingleStep(0.05)
        self.density_discount_spin.setValue(float(os.environ.get("DENSITY_DISCOUNT_FACTOR", "0.3")))
        self.density_discount_spin.setToolTip(
            "Discount factor for dense regions (0 = no effect, 1 = max discount).\n"
            "Short docs: 0.20 | Long docs: 0.35"
        )
        idc_form.addRow("  Density discount", self.density_discount_spin)

        # Intent Generation section
        intent_label = QLabel("<b>Intent Generation</b>")
        idc_form.addRow(intent_label)

        self.auto_adapt_intents_chk = QCheckBox("Auto-adapt intent generation (recommended for long docs)")
        self.auto_adapt_intents_chk.setChecked(os.environ.get("AUTO_ADAPT_INTENTS", "0") == "1")
        self.auto_adapt_intents_chk.setToolTip(
            "Automatically scale intent count based on document length.\n"
            "Essential for documents >400 sentences.\n"
            "Example: 495 sentences → 37 intents (vs default 15)\n"
            "Enabled by default for arXiv dataset."
        )
        idc_form.addRow(self.auto_adapt_intents_chk)

        # Force regeneration section
        force_label = QLabel("<b>Force Regeneration</b>")
        idc_form.addRow(force_label)

        self.force_intents_chk = QCheckBox("Force regenerate intents")
        self.force_intents_chk.setChecked(os.environ.get("FORCE_INTENTS", "0") == "1")
        self.force_intents_chk.setToolTip(
            "Skip cached intents and regenerate from scratch.\n"
            "Use when changing intent parameters or document."
        )
        idc_form.addRow(self.force_intents_chk)

        self.force_segments_chk = QCheckBox("Force regenerate IDC segments")
        self.force_segments_chk.setChecked(os.environ.get("FORCE_IDC_SEGMENTS", "0") == "1")
        self.force_segments_chk.setToolTip(
            "Skip cached IDC segments and regenerate from scratch.\n"
            "Use when changing IDC parameters."
        )
        idc_form.addRow(self.force_segments_chk)

        self.force_spans_chk = QCheckBox("Force regenerate evaluation spans")
        self.force_spans_chk.setChecked(os.environ.get("FORCE_SPANS", "0") == "1")
        self.force_spans_chk.setToolTip(
            "Skip cached evaluation spans and regenerate from scratch.\n"
            "Use when changing intent or document."
        )
        idc_form.addRow(self.force_spans_chk)

        idc_group.setLayout(idc_form)
        self.idc_group = idc_group
        self.idc_group.setVisible(False)

        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.load_btn = QPushButton("Load stats.json…")
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.setToolTip("Clear logs, charts, and loaded data")

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)

        # Charts - New comprehensive view
        self.charts_panel = ComprehensiveChartsPanel()
        
        # Create responsive scrollable area for charts
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.charts_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Optimize scrolling - prefer vertical over horizontal
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Layout
        cfg_box = QGroupBox("Configuration")
        form = QFormLayout()
        form.addRow("Dataset", self.dataset_combo_wrap)
        # Use explicit label widgets so we can reliably show/hide and retitle them
        self.doc_name_label_widget = QLabel("Document Name")
        form.addRow(self.doc_name_label_widget, self.doc_name_edit)
        self.limit_label_widget = QLabel("Documents to process")
        form.addRow(self.limit_label_widget, self.limit_spin)
        self.limit_label_widget.setVisible(False)
        self.dataset_path_label_widget = QLabel("Source JSON file")
        form.addRow(self.dataset_path_label_widget, self.ds_wrap)
        variants_layout = QHBoxLayout()
        variants_layout.addWidget(self.chk_idc)
        variants_layout.addWidget(self.chk_fixed)
        variants_layout.addWidget(self.chk_sliding)
        variants_layout.addWidget(self.chk_coh)
        variants_layout.addWidget(self.chk_para)
        v_wrap = QWidget(); v_wrap.setLayout(variants_layout)
        form.addRow("Variants", v_wrap)
        form.addRow(self.idc_group)

        btns = QHBoxLayout()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        btns.addWidget(self.clear_btn)
        btns.addWidget(self.load_btn)
        btns.addWidget(self.export_btn)
        btn_wrap = QWidget(); btn_wrap.setLayout(btns)
        form.addRow(btn_wrap)
        cfg_box.setLayout(form)
        cfg_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(cfg_box)
        config_layout.addStretch(1)
        config_panel.setMinimumWidth(420)

        config_scroll = QScrollArea()
        config_scroll.setWidget(config_panel)
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        config_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        config_scroll.setMinimumWidth(0)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(config_scroll, 1)
        left_layout.addWidget(QLabel("Logs"))
        left_layout.addWidget(self.log_edit, 1)
        left.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(scroll_area)
        # Set proportional sizes rather than fixed pixel values for responsiveness
        # Left panel: ~28%, Right panel: ~72% to avoid cramped controls
        total_width = self.width()
        left_width = int(total_width * 0.28)
        right_width = total_width - left_width
        splitter.setSizes([max(360, left_width), max(800, right_width)])

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.addWidget(splitter)
        self.setCentralWidget(container)

        # Process
        self.proc: Optional[QProcess] = None
        self.run_btn.clicked.connect(self.start_run)
        self.stop_btn.clicked.connect(self.stop_run)
        self.load_btn.clicked.connect(self.load_stats_dialog)
        self.export_btn.clicked.connect(self.export_csv)
        self.clear_btn.clicked.connect(self.clear_results)
        self.selected_dataset: Optional[DatasetInfo] = None
        self._on_dataset_changed()
        
        # Store current data for export
        self.current_variants = []
        self.current_metrics_data = {}
        
        # Store splitter reference for responsive resizing
        self.main_splitter = splitter

        # Menu
        file_menu = self.menuBar().addMenu("File")
        act_open = QAction("Load stats.json…", self)
        act_open.triggered.connect(self.load_stats_dialog)
        file_menu.addAction(act_open)
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

    def _current_dataset_info(self) -> Optional[DatasetInfo]:
        return self.dataset_info_by_label.get(self.dataset_combo.currentText())

    def _refresh_dataset_list(self) -> None:
        current_label = self.dataset_combo.currentText()
        self.dataset_catalog = self._build_dataset_catalog()
        self.dataset_info_by_label = {info.label: info for info in self.dataset_catalog}
        self._refresh_dataset_items(current_label)
        self._on_dataset_changed()

    def _build_dataset_catalog(self) -> List[DatasetInfo]:
        catalog: List[DatasetInfo] = []
        catalog.append(
            DatasetInfo(
                label="SQuAD2.0 dev",
                script=REPO_ROOT / "scripts" / "run_squad2_e2e.sh",
                dataset_type="corpus",
                default_path=REPO_ROOT / "data" / "squad" / "dev-v2.0.json",
                allow_browse=False,
                limit_applicable=True,
                enable_idc_params=True,
                stats_subdir="squad",
                description="Stanford Question Answering Dataset (dev set)",
            )
        )
        catalog.append(
            DatasetInfo(
                label="Qasper dev",
                script=REPO_ROOT / "scripts" / "run_qasper_e2e.sh",
                dataset_type="corpus",
                default_path=REPO_ROOT / "data" / "qasper" / "qasper-dev-v0.3.json",
                allow_browse=True,
                limit_applicable=True,
                enable_idc_params=True,
                path_env="QASPER_PATH",
                stats_subdir="qasper",
                description="Official Qasper dev split (information-seeking QA over research papers)",
            )
        )
        catalog.append(
            DatasetInfo(
                label="NewsQA corpus (concatenated)",
                script=REPO_ROOT / "scripts" / "run_idc_pipeline.sh",
                dataset_type="idc-corpus",
                default_path=REPO_ROOT / "data" / "input" / "newsqa_corpus.txt",
                allow_browse=True,
                limit_applicable=False,
                enable_idc_params=True,
                doc_name_visible=True,
                description="NewsQA: 10 stories concatenated into single corpus (~344 sentences, 66 questions)",
            )
        )
        fiori_path = REPO_ROOT / "data" / "fiori" / "fiori_tools_docs.txt"
        if fiori_path.exists():
            catalog.append(
                DatasetInfo(
                    label="Fiori Tools bundle",
                    script=REPO_ROOT / "scripts" / "run_idc_pipeline.sh",
                    dataset_type="idc-single",
                    default_path=fiori_path,
                    allow_browse=True,
                    limit_applicable=False,
                    enable_idc_params=True,
                    doc_name_visible=True,
                    path_env="INPUT_FILE",
                    description="Pre-packaged Fiori Tools documentation bundle",
                )
            )

        # Add arxiv_long dataset - 3 longest arXiv papers for testing long documents
        arxiv_long_path = REPO_ROOT / "data" / "arxiv_long" / "arxiv_bert_finance.txt"
        if arxiv_long_path.exists():
            catalog.append(
                DatasetInfo(
                    label="arXiv Long (3 papers, 1030 sentences)",
                    script=REPO_ROOT / "scripts" / "run_idc_pipeline.sh",
                    dataset_type="idc-single",
                    default_path=arxiv_long_path,
                    allow_browse=True,
                    limit_applicable=False,
                    enable_idc_params=True,
                    doc_name_visible=True,
                    path_env="INPUT_FILE",
                    description="Long arXiv papers (bert_finance: 495 sent, neural_nlp: 303 sent, transformers: 238 sent) - Tests IDC advantages on longer documents",
                )
            )

        # intentionally skip auto-listing every file in data/input/ to keep UI concise
        return catalog

    def _dataset_tooltip(self, info: DatasetInfo) -> str:
        parts = [info.label]
        if info.description:
            parts.append(info.description)
        if info.default_path:
            parts.append(str(info.default_path))
        return "\n".join(parts)

    def _refresh_dataset_items(self, selected_label: Optional[str] = None) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for info in self.dataset_catalog:
            self.dataset_combo.addItem(info.label)
            idx = self.dataset_combo.count() - 1
            self.dataset_combo.setItemData(idx, self._dataset_tooltip(info), Qt.ToolTipRole)
        if selected_label and selected_label in self.dataset_info_by_label:
            self.dataset_combo.setCurrentText(selected_label)
        elif self.dataset_catalog:
            self.dataset_combo.setCurrentIndex(0)
        self.dataset_combo.blockSignals(False)

    def _apply_dataset_info(self, info: DatasetInfo) -> None:
        self.selected_dataset = info
        self.dataset_combo.setToolTip(self._dataset_tooltip(info))
        self.idc_group.setVisible(info.enable_idc_params)
        self.dataset_path_label_widget.setText("Input .txt file" if info.dataset_type.startswith("idc") else "Source JSON file")
        tooltip = info.description or ("Select dataset file" if not info.dataset_type.startswith("idc") else "Select the document to process")
        self.dataset_path_edit.setToolTip(tooltip)

        if info.default_path:
            self.dataset_path_edit.setText(str(info.default_path))
        elif info.dataset_type.startswith("idc"):
            self.dataset_path_edit.clear()

        self.dataset_path_edit.setEnabled(info.allow_browse)
        self.browse_btn.setEnabled(info.allow_browse)

        self.doc_name_edit.setVisible(info.doc_name_visible)
        self.doc_name_label_widget.setVisible(info.doc_name_visible)
        if info.doc_name_visible:
            if info.default_path:
                self.doc_name_edit.setText(info.default_path.stem)
                self._last_auto_doc_name = info.default_path.stem
            elif not self.doc_name_edit.text().strip():
                self.doc_name_edit.setText("idc")
                self._last_auto_doc_name = "idc"
        else:
            self.doc_name_edit.clear()
            self._last_auto_doc_name = ""

        self.limit_spin.setVisible(info.limit_applicable)
        self.limit_label_widget.setVisible(info.limit_applicable)
        if not info.limit_applicable:
            self.limit_spin.setValue(int(os.environ.get("DEFAULT_DOC_LIMIT", "100")))

        # Set optimal defaults based on dataset type
        self._set_optimal_defaults(info)


    def _on_dataset_path_changed(self, text: str) -> None:
        info = self._current_dataset_info()
        if not info or not info.dataset_type.startswith("idc") or not info.doc_name_visible:
            return
        stem = Path(text).stem if text else ""
        if not stem:
            return
        current = self.doc_name_edit.text().strip()
        if not current or current == self._last_auto_doc_name:
            self._last_auto_doc_name = stem
            self.doc_name_edit.setText(stem)

    def resizeEvent(self, event):
        """Handle window resize to maintain proportional splitter sizes"""
        super().resizeEvent(event)
        if hasattr(self, 'main_splitter') and self.main_splitter:
            # Update splitter sizes proportionally on window resize
            total_width = event.size().width()
            left_width = int(total_width * 0.28)
            right_width = total_width - left_width
            self.main_splitter.setSizes([max(360, left_width), max(800, right_width)])

    def append_log(self, text: str):
        self.log_edit.appendPlainText(text.rstrip())
        self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def start_run(self):
        info = self._current_dataset_info()
        if info is None:
            QMessageBox.warning(self, "No dataset", "Select a dataset to run.")
            return
        script_path = info.script
        if not script_path.exists():
            QMessageBox.critical(self, "Script not found", f"Missing: {script_path}")
            return
        if self.proc is not None:
            QMessageBox.warning(self, "Already running", "A run is already in progress.")
            return

        variants = []
        if self.chk_idc.isChecked():
            variants.append("idc")
        if self.chk_fixed.isChecked():
            variants.append("fixed")
        if self.chk_sliding.isChecked():
            variants.append("sliding")
        if self.chk_coh.isChecked():
            variants.append("coh")
        if self.chk_para.isChecked():
            variants.append("paragraphs")
        if not variants:
            QMessageBox.warning(self, "No variants", "Select at least one variant to run.")
            return

        dataset_path = self.dataset_path_edit.text().strip()
        if not dataset_path and info.default_path is not None:
            dataset_path = str(info.default_path)
            self.dataset_path_edit.setText(dataset_path)
        if info.path_env and not dataset_path:
            QMessageBox.warning(self, "Missing input", "Please choose an input file for the selected dataset.")
            return
        if dataset_path and info.dataset_type.startswith("idc"):
            path_obj = Path(dataset_path)
            if not path_obj.exists() or not path_obj.is_file():
                QMessageBox.critical(self, "Invalid file", f"File does not exist: {dataset_path}")
                return

        self.log_edit.clear()
        self.append_log(f"→ Starting: {script_path}")
        chosen_dataset_path = dataset_path or (str(info.default_path) if info.default_path else "")
        summary_parts = [f"Dataset: {info.label}"]
        if chosen_dataset_path:
            summary_parts.append(f"path={chosen_dataset_path}")
        summary_parts.append(f"variants={'/'.join(variants)}")
        self.append_log("   " + " | ".join(summary_parts))

        env = os.environ.copy()
        env["VARIANTS"] = " ".join(variants)

        if info.limit_applicable:
            limit_value = str(self.limit_spin.value())
            env["LIMIT"] = limit_value
            env["DEFAULT_DOC_LIMIT"] = limit_value

        # Force regeneration flags (from checkboxes)
        env["FORCE_SENTENCES"] = "1"
        env["FORCE_INTENTS"] = "1" if self.force_intents_chk.isChecked() else "0"
        env["FORCE_EMBEDDINGS"] = "1"
        env["FORCE_IDC_SEGMENTS"] = "1" if self.force_segments_chk.isChecked() else "0"
        env["FORCE_CHUNK_EMBEDS"] = "1"
        env["FORCE_SPANS"] = "1" if self.force_spans_chk.isChecked() else "0"

        # IDC parameters
        env["LAMBDA"] = f"{self.lambda_spin.value():.6f}"
        env["MAX_LEN"] = str(self.max_len_spin.value())
        env["MIN_LEN"] = str(self.min_len_spin.value())
        env["BOUNDARY_PENALTY"] = f"{self.boundary_penalty_spin.value():.2f}"
        env["COHERENCE_WEIGHT"] = f"{self.coherence_weight_spin.value():.2f}"
        env["AUTO_TUNE"] = "1" if self.auto_tune_chk.isChecked() else "0"
        env["AUTO_TUNE_BASELINES"] = env["AUTO_TUNE"]

        # Intent generation
        env["AUTO_ADAPT_INTENTS"] = "1" if self.auto_adapt_intents_chk.isChecked() else "0"

        # Advanced optimization parameters
        env["CONTEXTUAL_EMBEDDINGS"] = "1" if self.contextual_emb_chk.isChecked() else "0"
        env["CONTEXT_WEIGHT"] = f"{self.context_weight_spin.value():.2f}"
        env["DENSITY_AWARE"] = "1" if self.density_aware_chk.isChecked() else "0"
        env["DENSITY_DISCOUNT_FACTOR"] = f"{self.density_discount_spin.value():.2f}"

        if info.path_env and dataset_path:
            env[info.path_env] = dataset_path
        if info.dataset_type.startswith("idc") and dataset_path:
            doc_name = self.doc_name_edit.text().strip()
            if not doc_name:
                doc_name = Path(dataset_path).stem
                self.doc_name_edit.setText(doc_name)
            env["DOC_NAME"] = doc_name

        self.proc = QProcess(self)
        self.proc.setProgram("bash")
        self.proc.setArguments([str(script_path)])
        self.proc.setProcessEnvironment(self._env_to_qprocess(env))
        self.proc.setWorkingDirectory(str(REPO_ROOT))
        self.proc.readyReadStandardOutput.connect(lambda: self._read_stream(self.proc.readAllStandardOutput().data().decode("utf-8", errors="ignore")))
        self.proc.readyReadStandardError.connect(lambda: self._read_stream(self.proc.readAllStandardError().data().decode("utf-8", errors="ignore")))
        self.proc.finished.connect(self._on_finished)
        self.proc.start()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_run(self):
        if self.proc is not None:
            self.append_log("→ Stopping…")
            self.proc.kill()
            self.proc = None
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def _read_stream(self, s: str):
        if s:
            self.append_log(s)

    def _on_finished(self, code: int, status):
        self.append_log(f"→ Finished with code {code}")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.proc = None

        info = self._current_dataset_info()
        stats_path: Optional[Path] = None
        if info and info.dataset_type.startswith("idc"):
            dataset_path = self.dataset_path_edit.text().strip()
            if not dataset_path and info.default_path is not None:
                dataset_path = str(info.default_path)
            doc_name = Path(dataset_path).stem if dataset_path else "idc"
            stats_path = REPO_ROOT / "out" / doc_name / "stats.json"
        elif info and info.stats_subdir:
            stats_path = REPO_ROOT / "out" / info.stats_subdir / "stats.json"

        if stats_path and stats_path.exists():
            self.load_stats(stats_path)

    def _env_to_qprocess(self, env: Dict[str, str]):
        from PySide6.QtCore import QProcessEnvironment
        pe = QProcessEnvironment.systemEnvironment()
        for k, v in env.items():
            pe.insert(k, v)
        return pe

    def load_stats_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open stats.json", str(REPO_ROOT / "out"), "JSON files (*.json)")
        if path:
            self.load_stats(Path(path))

    def _set_optimal_defaults(self, info: DatasetInfo) -> None:
        """Set optimal default parameters based on dataset type."""
        # Determine dataset name from label
        dataset_label = info.label.lower()

        if "squad" in dataset_label:
            # SQuAD: Enable auto-tuning, no auto-adapt needed (short docs ~200 sent)
            self.auto_tune_chk.setChecked(True)
            self.auto_adapt_intents_chk.setChecked(False)
            self.force_intents_chk.setChecked(True)
            self.force_segments_chk.setChecked(True)
            self.force_spans_chk.setChecked(True)
            self.contextual_emb_chk.setChecked(True)
            self.density_aware_chk.setChecked(True)
            # Optimal manual parameters (if auto-tune disabled)
            self.lambda_spin.setValue(0.02)  # Best for SQuAD based on analysis
            self.boundary_penalty_spin.setValue(1.2)
            self.max_len_spin.setValue(20)
            self.coherence_weight_spin.setValue(0.1)
            # Process all docs in SQuAD dev set
            if info.limit_applicable:
                self.limit_spin.setValue(100)

        elif "arxiv" in dataset_label:
            # arXiv: Enable auto-adaptation (long docs ~495 sent), force regeneration
            self.auto_tune_chk.setChecked(False)  # Not needed for arXiv
            self.auto_adapt_intents_chk.setChecked(True)  # CRITICAL for arXiv
            self.force_intents_chk.setChecked(True)
            self.force_segments_chk.setChecked(True)
            self.force_spans_chk.setChecked(True)
            self.contextual_emb_chk.setChecked(True)
            self.density_aware_chk.setChecked(True)
            # Optimal parameters for long research papers
            self.lambda_spin.setValue(0.0005)  # Very low for long, intent-driven chunks
            self.boundary_penalty_spin.setValue(0.8)
            self.max_len_spin.setValue(15)
            self.coherence_weight_spin.setValue(0.15)

        elif "fiori" in dataset_label:
            # Fiori: Default settings work well (short technical docs)
            self.auto_tune_chk.setChecked(False)
            self.auto_adapt_intents_chk.setChecked(False)
            self.force_intents_chk.setChecked(True)
            self.force_segments_chk.setChecked(True)
            self.force_spans_chk.setChecked(True)
            self.contextual_emb_chk.setChecked(True)
            self.density_aware_chk.setChecked(True)
            # Optimal parameters for short technical documentation
            self.lambda_spin.setValue(0.015)  # Medium chunks, paragraph-like
            self.boundary_penalty_spin.setValue(1.0)
            self.max_len_spin.setValue(10)
            self.coherence_weight_spin.setValue(0.1)

        elif "newsqa" in dataset_label and "corpus" in dataset_label:
            # NewsQA Corpus: Concatenated single document (~344 sentences)
            self.auto_tune_chk.setChecked(False)
            self.auto_adapt_intents_chk.setChecked(False)  # 344 sentences doesn't need auto-adapt
            self.force_intents_chk.setChecked(True)
            self.force_segments_chk.setChecked(True)
            self.force_spans_chk.setChecked(True)
            self.contextual_emb_chk.setChecked(True)
            self.density_aware_chk.setChecked(True)
            # Optimal parameters for medium-length news corpus
            self.lambda_spin.setValue(0.01)  # Medium chunks for Q&A
            self.boundary_penalty_spin.setValue(1.0)
            self.max_len_spin.setValue(14)
            self.coherence_weight_spin.setValue(0.1)
            # Pre-set doc_name if field is visible
            if info.doc_name_visible and not self.doc_name_edit.text().strip():
                self.doc_name_edit.setText("newsqa_corpus")

        elif "qasper" in dataset_label:
            # Qasper: Research papers with varying lengths
            # Similar to SQuAD but for academic papers
            self.auto_tune_chk.setChecked(False)  # Can enable if many papers evaluated
            self.auto_adapt_intents_chk.setChecked(True)  # Papers can be long
            self.force_intents_chk.setChecked(True)
            self.force_segments_chk.setChecked(True)
            self.force_spans_chk.setChecked(True)
            self.contextual_emb_chk.setChecked(True)
            self.density_aware_chk.setChecked(True)
            # Optimal parameters for research papers
            self.lambda_spin.setValue(0.002)  # Balanced for varying paper lengths
            self.boundary_penalty_spin.setValue(1.0)
            self.max_len_spin.setValue(15)
            self.coherence_weight_spin.setValue(0.15)
            # Default to reasonable sample size for testing
            if info.limit_applicable:
                self.limit_spin.setValue(10)

        else:
            # Other datasets: Conservative defaults
            self.auto_tune_chk.setChecked(False)
            self.auto_adapt_intents_chk.setChecked(False)
            self.force_intents_chk.setChecked(False)
            self.force_segments_chk.setChecked(False)
            self.force_spans_chk.setChecked(False)
            self.contextual_emb_chk.setChecked(True)
            self.density_aware_chk.setChecked(True)
            # Conservative default parameters (medium chunks)
            self.lambda_spin.setValue(0.01)
            self.boundary_penalty_spin.setValue(1.0)
            self.max_len_spin.setValue(12)
            self.coherence_weight_spin.setValue(0.1)
            # Default to 10 documents if limit applicable
            if info.limit_applicable:
                self.limit_spin.setValue(10)

    def _on_dataset_changed(self):
        info = self._current_dataset_info()
        if info is None:
            return
        self._apply_dataset_info(info)

    def _browse_dataset(self):
        info = self._current_dataset_info()
        if info is None:
            return
        if info.dataset_type.startswith("idc"):
            default_dir = str((Path(self.dataset_path_edit.text()).parent if self.dataset_path_edit.text().strip() else REPO_ROOT / "data" / "input").resolve())
            path, _ = QFileDialog.getOpenFileName(self, "Select document", default_dir, "Text files (*.txt);;All files (*)")
        else:
            default_dir = str((Path(self.dataset_path_edit.text()).parent if self.dataset_path_edit.text().strip() else REPO_ROOT / "data").resolve())
            path, _ = QFileDialog.getOpenFileName(self, "Select dataset file", default_dir, "JSON files (*.json);;All files (*)")
        if path:
            self.dataset_path_edit.setText(path)
            if info.dataset_type.startswith("idc"):
                self.doc_name_edit.setText(Path(path).stem)

    def load_stats(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Failed to load", f"Could not read {path}:\n{e}")
            return
        results = data.get("results", [])
        
        # Build maps
        variants_order = []
        metrics_data = {
            "r1": {},
            "r5": {},
            "mrr": {},
            "cov": {},
            "completeness": {},  # NEW: Extended metrics
            "diversity": {},
            "redundancy": {},
            "efficiency": {},
            "len": {},
            "tok": {}
        }
        
        for r in results:
            if not r.get("available", False):
                continue
            var = r.get("variant")
            if var is None: 
                continue
            variants_order.append(var)
            
            # span_mode metrics
            sm = r.get("span_mode", {})
            def mean_of(metric):
                x = sm.get(metric)
                if isinstance(x, dict):
                    return float(x.get("mean", 0.0))
                return None
            
            metrics_data["r1"][var] = mean_of("R1") or 0.0
            metrics_data["r5"][var] = mean_of("RK") or 0.0  
            metrics_data["mrr"][var] = mean_of("MRR") or 0.0
            
            # Coverage
            cov = r.get("coverage", {})
            metrics_data["cov"][var] = float(cov.get("mean", 0.0)) if isinstance(cov, dict) else 0.0

            # NEW: Extended metrics
            ext = r.get("extended_metrics", {})
            def ext_mean_of(metric):
                x = ext.get(metric, {})
                if isinstance(x, dict):
                    return float(x.get("mean", 0.0))
                return 0.0

            metrics_data["completeness"][var] = ext_mean_of("completeness")
            metrics_data["diversity"][var] = ext_mean_of("diversity")
            metrics_data["redundancy"][var] = ext_mean_of("redundancy")
            metrics_data["efficiency"][var] = ext_mean_of("efficiency")

            # Average length
            metrics_data["len"][var] = float(r.get("avg_sentences_per_chunk", 0.0))
            metrics_data["tok"][var] = float(r.get("avg_tokens_per_chunk", 0.0))

        # Deduplicate variants while preserving order
        seen = set()
        variants = []
        for v in variants_order:
            if v not in seen:
                seen.add(v)
                variants.append(v)
                
        if not variants:
            QMessageBox.information(self, "No results", "No available variants to chart")
            return

        # Auto-adjust length chart scale
        max_len = max([metrics_data["len"].get(v, 0.0) for v in variants] + [10.0])
        if "len" in self.charts_panel.charts:
            chart_panel, _ = self.charts_panel.charts["len"]
            self.charts_panel.charts["len"] = (chart_panel, max_len * 1.2)

        max_tok = max([metrics_data["tok"].get(v, 0.0) for v in variants] + [100.0])
        if "tok" in self.charts_panel.charts:
            chart_panel, _ = self.charts_panel.charts["tok"]
            self.charts_panel.charts["tok"] = (chart_panel, max_tok * 1.2)

        # Store data for export
        self.current_variants = variants
        self.current_metrics_data = metrics_data
        self.export_btn.setEnabled(True)
        
        # Update all charts at once
        self.charts_panel.update_charts(variants, metrics_data)
        
        # Show summary in log
        self.append_log(f"📊 Loaded results for {len(variants)} variants: {', '.join(variants)}")
        
        # Print key metrics summary
        best_variant = None
        best_combined_score = 0.0
        for variant in variants:
            r1 = metrics_data["r1"].get(variant, 0.0)
            mrr = metrics_data["mrr"].get(variant, 0.0) 
            cov = metrics_data["cov"].get(variant, 0.0)
            combined = (r1 + mrr + cov) / 3  # Simple average
            if combined > best_combined_score:
                best_combined_score = combined
                best_variant = variant
                
        if best_variant:
            r1 = metrics_data["r1"][best_variant]
            mrr = metrics_data["mrr"][best_variant]
            cov = metrics_data["cov"][best_variant]
            avg_len = metrics_data["len"][best_variant]
            self.append_log(f"🏆 Best variant: {best_variant} (R@1={r1:.3f}, MRR={mrr:.3f}, Cov={cov:.3f}, AvgLen={avg_len:.1f})")
        
        # Run chunk quality analysis if possible
        self.run_chunk_quality_analysis()
    
    def export_csv(self):
        """Export current results to CSV file"""
        if not self.current_variants:
            QMessageBox.information(self, "No Data", "No results to export. Load stats.json first.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", 
            str(REPO_ROOT / "out" / "results_export.csv"),
            "CSV files (*.csv)"
        )
        if not path:
            return
            
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header (including extended metrics)
                writer.writerow(['Variant', 'R@1', 'R@5', 'MRR', 'Coverage', 'Completeness', 'Diversity', 'Redundancy', 'Efficiency', 'Avg Length', 'Avg Tokens'])

                # Write data
                for variant in self.current_variants:
                    row = [
                        variant,
                        f"{self.current_metrics_data['r1'].get(variant, 0.0):.3f}",
                        f"{self.current_metrics_data['r5'].get(variant, 0.0):.3f}",
                        f"{self.current_metrics_data['mrr'].get(variant, 0.0):.3f}",
                        f"{self.current_metrics_data['cov'].get(variant, 0.0):.3f}",
                        f"{self.current_metrics_data['completeness'].get(variant, 0.0):.3f}",  # NEW
                        f"{self.current_metrics_data['diversity'].get(variant, 0.0):.3f}",  # NEW
                        f"{self.current_metrics_data['redundancy'].get(variant, 0.0):.3f}",  # NEW
                        f"{self.current_metrics_data['efficiency'].get(variant, 0.0):.3f}",  # NEW
                        f"{self.current_metrics_data['len'].get(variant, 0.0):.1f}",
                        f"{self.current_metrics_data['tok'].get(variant, 0.0):.1f}"
                    ]
                    writer.writerow(row)
                    
            self.append_log(f"📄 Exported results to {path}")
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export CSV:\n{e}")
    
    def clear_results(self):
        """Clear all results, logs, charts, and loaded data"""
        reply = QMessageBox.question(
            self, "Clear Results", 
            "This will clear all logs, charts, and loaded data. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear logs
            self.log_edit.clear()
            self.append_log("Results cleared.")
            
            # Clear current data
            self.current_variants = []
            self.current_metrics_data = {}
            
            # Disable export button
            self.export_btn.setEnabled(False)
            
            # Clear all charts directly using the internal method
            self.charts_panel._clear_all_charts()
            
            self.append_log("📄 All charts, logs, and data have been cleared.")
    
    def run_chunk_quality_analysis(self):
        """Run chunk quality analysis and display results"""
        try:
            from pathlib import Path
            import subprocess
            import sys
            
            # Find current output directory
            output_dir = Path(self.output_dir.text().strip()) if self.output_dir.text().strip() else REPO_ROOT / "out"
            
            # Check if chunk files exist
            chunk_files = list(output_dir.glob("chunks.*.jsonl"))
            if not chunk_files:
                return  # Silently skip if no chunks found
            
            # Run the chunk quality analysis script
            script_path = REPO_ROOT / "scripts" / "analyze_chunk_quality.py"
            if not script_path.exists():
                return
            
            self.append_log("📈 Running chunk quality analysis...")
            
            # Change to the output directory and run analysis
            result = subprocess.run([
                sys.executable, str(script_path)
            ], cwd=output_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse and display key insights from output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if '📊' in line or '💡' in line or '🏆' in line or '📂' in line:
                        self.append_log(line)
                
                # Check if quality charts were generated
                charts_dir = output_dir / "quality_analysis"
                if charts_dir.exists():
                    chart_files = list(charts_dir.glob("*.png"))
                    if chart_files:
                        self.append_log(f"📊 Generated {len(chart_files)} chunk quality charts in {charts_dir}")
                        self.append_log("   • answer_chunk_lengths.png - Average length of answer-containing chunks")
                        self.append_log("   • answer_precision_ratios.png - Answer-to-noise ratio in chunks") 
                        self.append_log("   • answer_padding.png - Padding around answers")
                        self.append_log("   • chunk_length_distributions.png - Length distribution analysis")
            
        except Exception:
            # Silently handle errors - this is supplementary analysis
            pass


def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
