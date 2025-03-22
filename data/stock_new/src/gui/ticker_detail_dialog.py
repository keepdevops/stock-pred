from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QLabel, QTableWidget, QTableWidgetItem, QPushButton,
                           QWidget, QGridLayout, QFrame, QScrollArea)
from PyQt5.QtCore import Qt
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

class TickerDetailDialog(QDialog):
    def __init__(self, ticker_symbol, parent=None):
        super().__init__(parent)
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.setWindowTitle(f"Ticker Details - {ticker_symbol}")
        self.setGeometry(100, 100, 1000, 800)
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Overview tab
        self.overview_tab = QWidget()
        self.setup_overview_tab()
        self.tabs.addTab(self.overview_tab, "Overview")
        
        # Financials tab
        self.financials_tab = QWidget()
        self.setup_financials_tab()
        self.tabs.addTab(self.financials_tab, "Financials")
        
        # Chart tab
        self.chart_tab = QWidget()
        self.setup_chart_tab()
        self.tabs.addTab(self.chart_tab, "Chart")
        
        # News tab
        self.news_tab = QWidget()
        self.setup_news_tab()
        self.tabs.addTab(self.news_tab, "News")
        
        layout.addWidget(self.tabs)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def setup_overview_tab(self):
        layout = QVBoxLayout(self.overview_tab)
        
        # Company info section
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.StyledPanel)
        info_layout = QGridLayout(info_frame)
        
        self.info_labels = {
            'Company Name': QLabel(),
            'Sector': QLabel(),
            'Industry': QLabel(),
            'Market Cap': QLabel(),
            'Current Price': QLabel(),
            'PE Ratio': QLabel(),
            'Forward PE': QLabel(),
            'PEG Ratio': QLabel(),
            'Dividend Yield': QLabel(),
            'Beta': QLabel(),
            '52 Week High': QLabel(),
            '52 Week Low': QLabel(),
            'Volume': QLabel(),
            'Avg Volume': QLabel()
        }
        
        row = 0
        col = 0
        for label, widget in self.info_labels.items():
            info_layout.addWidget(QLabel(f"{label}:"), row, col * 2)
            info_layout.addWidget(widget, row, col * 2 + 1)
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        layout.addWidget(info_frame)
        
        # Description section
        desc_frame = QFrame()
        desc_frame.setFrameStyle(QFrame.StyledPanel)
        desc_layout = QVBoxLayout(desc_frame)
        desc_layout.addWidget(QLabel("Business Description:"))
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        desc_layout.addWidget(self.description_label)
        
        layout.addWidget(desc_frame)

    def setup_financials_tab(self):
        layout = QVBoxLayout(self.financials_tab)
        
        # Financial tables
        self.income_table = QTableWidget()
        self.balance_table = QTableWidget()
        self.cash_flow_table = QTableWidget()
        
        # Add tables to tab
        financial_tabs = QTabWidget()
        financial_tabs.addTab(self.income_table, "Income Statement")
        financial_tabs.addTab(self.balance_table, "Balance Sheet")
        financial_tabs.addTab(self.cash_flow_table, "Cash Flow")
        
        layout.addWidget(financial_tabs)

    def setup_chart_tab(self):
        layout = QVBoxLayout(self.chart_tab)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        # Period selection
        self.period_combo = QComboBox()
        self.period_combo.addItems(['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'])
        self.period_combo.setCurrentText('6mo')
        controls_layout.addWidget(QLabel("Period:"))
        controls_layout.addWidget(self.period_combo)
        
        # Interval selection
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'])
        self.interval_combo.setCurrentText('1d')
        controls_layout.addWidget(QLabel("Interval:"))
        controls_layout.addWidget(self.interval_combo)
        
        # Update button
        update_button = QPushButton("Update Chart")
        update_button.clicked.connect(self.update_chart)
        controls_layout.addWidget(update_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Chart container
        self.chart_container = QWidget()
        layout.addWidget(self.chart_container)

    def setup_news_tab(self):
        layout = QVBoxLayout(self.news_tab)
        
        # News table
        self.news_table = QTableWidget()
        self.news_table.setColumnCount(3)
        self.news_table.setHorizontalHeaderLabels(["Date", "Title", "Source"])
        self.news_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.news_table)

    def load_data(self):
        try:
            # Get ticker info
            info = self.ticker.info
            
            # Update overview tab
            self.info_labels['Company Name'].setText(info.get('longName', 'N/A'))
            self.info_labels['Sector'].setText(info.get('sector', 'N/A'))
            self.info_labels['Industry'].setText(info.get('industry', 'N/A'))
            self.info_labels['Market Cap'].setText(f"${info.get('marketCap', 0):,}")
            self.info_labels['Current Price'].setText(f"${info.get('currentPrice', 0):,.2f}")
            self.info_labels['PE Ratio'].setText(f"{info.get('trailingPE', 'N/A')}")
            self.info_labels['Forward PE'].setText(f"{info.get('forwardPE', 'N/A')}")
            self.info_labels['PEG Ratio'].setText(f"{info.get('pegRatio', 'N/A')}")
            self.info_labels['Dividend Yield'].setText(f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A')
            self.info_labels['Beta'].setText(f"{info.get('beta', 'N/A')}")
            self.info_labels['52 Week High'].setText(f"${info.get('fiftyTwoWeekHigh', 0):,.2f}")
            self.info_labels['52 Week Low'].setText(f"${info.get('fiftyTwoWeekLow', 0):,.2f}")
            self.info_labels['Volume'].setText(f"{info.get('volume', 0):,}")
            self.info_labels['Avg Volume'].setText(f"{info.get('averageVolume', 0):,}")
            
            self.description_label.setText(info.get('longBusinessSummary', 'No description available.'))
            
            # Load financials
            self.load_financials()
            
            # Load initial chart
            self.update_chart()
            
            # Load news
            self.load_news()
            
        except Exception as e:
            print(f"Error loading ticker data: {e}")

    def load_financials(self):
        try:
            # Income Statement
            income_stmt = self.ticker.financials
            self.populate_table(self.income_table, income_stmt, "Income Statement")
            
            # Balance Sheet
            balance_sheet = self.ticker.balance_sheet
            self.populate_table(self.balance_table, balance_sheet, "Balance Sheet")
            
            # Cash Flow
            cash_flow = self.ticker.cashflow
            self.populate_table(self.cash_flow_table, cash_flow, "Cash Flow")
            
        except Exception as e:
            print(f"Error loading financials: {e}")

    def populate_table(self, table, data, title):
        if data is None or data.empty:
            return
        
        table.setRowCount(len(data.index))
        table.setColumnCount(len(data.columns))
        
        # Set headers
        table.setHorizontalHeaderLabels([d.strftime('%Y-%m-%d') for d in data.columns])
        table.setVerticalHeaderLabels(data.index)
        
        # Populate data
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                value = data.iloc[i, j]
                if pd.isna(value):
                    item = QTableWidgetItem('N/A')
                else:
                    item = QTableWidgetItem(f"${value:,.0f}" if value else '0')
                table.setItem(i, j, item)
        
        table.resizeColumnsToContents()

    def update_chart(self):
        try:
            period = self.period_combo.currentText()
            interval = self.interval_combo.currentText()
            
            # Get historical data
            hist = self.ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return
            
            # Create figure with secondary y-axis
            fig = make_subplots(rows=2, cols=1, shared_xaxis=True, 
                              vertical_spacing=0.03, 
                              row_heights=[0.7, 0.3])

            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='OHLC'
            ), row=1, col=1)

            # Add volume bar chart
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume'
            ), row=2, col=1)

            # Update layout
            fig.update_layout(
                title=f'{self.ticker_symbol} Chart',
                yaxis_title='Price',
                yaxis2_title='Volume',
                xaxis_rangeslider_visible=False,
                height=600
            )

            # Save to HTML and display
            chart_path = f'temp_{self.ticker_symbol}_chart.html'
            fig.write_html(chart_path)
            
            # Load the chart in a WebView
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            web_view = QWebEngineView()
            web_view.load(QUrl.fromLocalFile(os.path.abspath(chart_path)))
            
            # Clear and update chart container
            for i in reversed(range(self.chart_container.layout().count())): 
                self.chart_container.layout().itemAt(i).widget().setParent(None)
            self.chart_container.layout().addWidget(web_view)
            
        except Exception as e:
            print(f"Error updating chart: {e}")

    def load_news(self):
        try:
            news = self.ticker.news
            self.news_table.setRowCount(len(news))
            
            for i, item in enumerate(news):
                date = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
                self.news_table.setItem(i, 0, QTableWidgetItem(date))
                self.news_table.setItem(i, 1, QTableWidgetItem(item['title']))
                self.news_table.setItem(i, 2, QTableWidgetItem(item['publisher']))
            
            self.news_table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"Error loading news: {e}") 