(stockmarket) moose@m3 yahoo-stock % python yahoo1.py
2025-02-12 18:45:26.605 python[15529:413772] +[IMKClient subclass]: chose IMKClient_Modern
2025-02-12 18:45:26.605 python[15529:413772] +[IMKInputSession subclass]: chose IMKInputSession_Modern
$TSLA: possibly delisted; no price data found  (period=1y)
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-news.py		yahoo1-nyse.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stocks.db		yahoo-rss.py		yahoo1.py
__pycache__		duckdb_data		forexduckdb-query.py	yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	yahoo-feedparser.py	yahoo.py
(stockmarket) moose@m3 yahoo-stock % du -a 
8	./yahoo-zmq.py
48	./forex-plot.py
104	./yahoo1.py
16	./data-yahoo-finance.py
16	./yahoo.py
8	./yahoo-data-news.py
24	./__pycache__/yahoo_base.cpython-39.pyc
24	./__pycache__
48	./forexdata.py
120	./yahoo1-nyse.py
40	./Cboedata.py
2072	./forex-duckdb.db
8	./yahoo-rss.py
59928	./stocks.db
8	./download_stats_a.txt
8	./yahoo-feedparser.py
536	./duckdb_data/default.db
0	./duckdb_data/forex-duckdb.db
0	./duckdb_data/stocks.db
0	./duckdb_data/nasdaq_stocks.db
536	./duckdb_data/Test.db
1072	./duckdb_data
8	./zmq-yahoo-client.py
8	./forexduckdb-query.py
12312	./nasdaq_stocks.db
40	./Nasqdata.py
8	./yahoo-news.py
120	./stock.db
40	./data.py
76064	.
(stockmarket) moose@m3 yahoo-stock % du -a skh
du: skh: No such file or directory
(stockmarket) moose@m3 yahoo-stock % du -a h
du: h: No such file or directory
(stockmarket) moose@m3 yahoo-stock % du -a shk
du: shk: No such file or directory
(stockmarket) moose@m3 yahoo-stock % du -a
8	./yahoo-zmq.py
48	./forex-plot.py
104	./yahoo1.py
16	./data-yahoo-finance.py
16	./yahoo.py
8	./yahoo-data-news.py
24	./__pycache__/yahoo_base.cpython-39.pyc
24	./__pycache__
48	./forexdata.py
120	./yahoo1-nyse.py
40	./Cboedata.py
2072	./forex-duckdb.db
8	./yahoo-rss.py
59928	./stocks.db
8	./download_stats_a.txt
8	./yahoo-feedparser.py
536	./duckdb_data/default.db
0	./duckdb_data/forex-duckdb.db
0	./duckdb_data/stocks.db
0	./duckdb_data/nasdaq_stocks.db
536	./duckdb_data/Test.db
1072	./duckdb_data
8	./zmq-yahoo-client.py
8	./forexduckdb-query.py
12312	./nasdaq_stocks.db
40	./Nasqdata.py
8	./yahoo-news.py
120	./stock.db
40	./data.py
76064	.
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-news.py		yahoo1-nyse.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stocks.db		yahoo-rss.py		yahoo1.py
__pycache__		duckdb_data		forexduckdb-query.py	yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	yahoo-feedparser.py	yahoo.py
(stockmarket) moose@m3 yahoo-stock % python forexd
python: can't open file '/Users/moose/stock-api/yahoo-stock/forexd': [Errno 2] No such file or directory
(stockmarket) moose@m3 yahoo-stock % python forexduckdb-query.py 
(stockmarket) moose@m3 yahoo-stock % vi forexduckdb-query.py
(stockmarket) moose@m3 yahoo-stock % vi duckdb_data
(stockmarket) moose@m3 yahoo-stock % cd duckdb_data
(stockmarket) moose@m3 duckdb_data % ls
Test.db			default.db		forex-duckdb.db		nasdaq_stocks.db	stocks.db
(stockmarket) moose@m3 duckdb_data % vi Test.db
(stockmarket) moose@m3 duckdb_data % du -a
536	./default.db
0	./forex-duckdb.db
0	./stocks.db
0	./nasdaq_stocks.db
536	./Test.db
1072	.
(stockmarket) moose@m3 duckdb_data % ls
Test.db			default.db		forex-duckdb.db		nasdaq_stocks.db	stocks.db
(stockmarket) moose@m3 duckdb_data % cd ..
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-news.py		yahoo1-nyse.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stocks.db		yahoo-rss.py		yahoo1.py
__pycache__		duckdb_data		forexduckdb-query.py	yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	yahoo-feedparser.py	yahoo.py
(stockmarket) moose@m3 yahoo-stock % vi forex-duckdb.db
(stockmarket) moose@m3 yahoo-stock % du -a
8	./yahoo-zmq.py
48	./forex-plot.py
104	./yahoo1.py
16	./data-yahoo-finance.py
16	./yahoo.py
8	./yahoo-data-news.py
24	./__pycache__/yahoo_base.cpython-39.pyc
24	./__pycache__
48	./forexdata.py
120	./yahoo1-nyse.py
40	./Cboedata.py
2072	./forex-duckdb.db
8	./yahoo-rss.py
59928	./stocks.db
8	./download_stats_a.txt
8	./yahoo-feedparser.py
536	./duckdb_data/default.db
0	./duckdb_data/forex-duckdb.db
0	./duckdb_data/stocks.db
0	./duckdb_data/nasdaq_stocks.db
536	./duckdb_data/Test.db
1072	./duckdb_data
8	./zmq-yahoo-client.py
8	./forexduckdb-query.py
12312	./nasdaq_stocks.db
40	./Nasqdata.py
8	./yahoo-news.py
120	./stock.db
40	./data.py
76064	.
(stockmarket) moose@m3 yahoo-stock % $ duckdb
v1.0.0 1f98600c2c
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database.
D

zsh: command not found: $
zsh: command not found: v1.0.0
zsh: command not found: Enter
zsh: command not found: Connected
zsh: command not found: Use
zsh: command not found: D
(stockmarket) moose@m3 yahoo-stock % $ duckdb
D CREATE TABLE ducks AS SELECT 3 AS age, 'mandarin' AS breed;
FROM ducks;
┌───────┬──────────┐
│  age  │  breed   │
│ int32 │ varchar  │
├───────┼──────────┤
│     3 │ mandarin │
└───────┴──────────┘

zsh: command not found: $
zsh: command not found: D
zsh: command not found: FROM
zsh: command not found: ┌───────┬──────────┐
zsh: command not found: │
zsh: command not found: │
zsh: command not found: ├───────┼──────────┤
zsh: command not found: │
zsh: command not found: └───────┴──────────┘
(stockmarket) moose@m3 yahoo-stock % import duckdb
import pyarrow as pa

arrow_table = pa.Table.from_pydict({"a": [42]})
duckdb.sql("SELECT * FROM arrow_table")
zsh: command not found: import
zsh: command not found: import
zsh: unknown file attribute: {
zsh: number expected
(stockmarket) moose@m3 yahoo-stock % import duckdb

duckdb.sql("SELECT 42").fetchall()   # Python objects
duckdb.sql("SELECT 42").df()         # Pandas DataFrame
duckdb.sql("SELECT 42").pl()         # Polars DataFrame
duckdb.sql("SELECT 42").arrow()      # Arrow Table
duckdb.sql("SELECT 42").fetchnumpy() # NumPy Arrays
zsh: command not found: import
zsh: no matches found: duckdb.sql(SELECT 42).fetchall
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-news.py		yahoo1-nyse.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stocks.db		yahoo-rss.py		yahoo1.py
__pycache__		duckdb_data		forexduckdb-query.py	yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	yahoo-feedparser.py	yahoo.py
(stockmarket) moose@m3 yahoo-stock % import duckdb

duckdb.sql("SELECT * ").fetchall()   # Python objects
duckdb.sql("SELECT * ").df()         # Pandas DataFrame
duckdb.sql("SELECT * ").pl()         # Polars DataFrame
duckdb.sql("SELECT * ").arrow()      # Arrow Table
duckdb.sql("SELECT * ").fetchnumpy() # NumPy Arrays
zsh: command not found: import
zsh: no matches found: duckdb.sql(SELECT * ).fetchall
(stockmarket) moose@m3 yahoo-stock %              
(stockmarket) moose@m3 yahoo-stock % import duckdb

duckdb.sql("SELECT ticker  ").fetchall()   # Python objects
duckdb.sql("SELECT ticker  ").df()         # Pandas DataFrame
duckdb.sql("SELECT ticker  ").pl()         # Polars DataFrame
duckdb.sql("SELECT ticker  ").arrow()      # Arrow Table
duckdb.sql("SELECT ticker  ").fetchnumpy() # NumPy Arrays
zsh: command not found: import
zsh: no matches found: duckdb.sql(SELECT ticker  ).fetchall
(stockmarket) moose@m3 yahoo-stock % import duckdb

con = duckdb.connect()
con.sql("SELECT 42 AS x").show()
function function> import duckdb

con = duckdb.connect()
con.sql("SELECT 42 AS x").show()
function function> 
(stockmarket) moose@m3 yahoo-stock % df = con.execute("SELECT * FROM items").fetchdf()
print(df)
zsh: no matches found: con.execute(SELECT * FROM items).fetchdf
(stockmarket) moose@m3 yahoo-stock % df = con.execute("SELECT * FROM items").fetchdf()
print(df)
zsh: no matches found: con.execute(SELECT * FROM items).fetchdf
(stockmarket) moose@m3 yahoo-stock % INSTALL arrow;
LOAD arrow;
usage: install [-bCcpSsUv] [-f flags] [-g group] [-m mode] [-o owner]
               [-M log] [-D dest] [-h hash] [-T tags]
               [-B suffix] [-l linkflags] [-N dbdir]
               file1 file2
       install [-bCcpSsUv] [-f flags] [-g group] [-m mode] [-o owner]
               [-M log] [-D dest] [-h hash] [-T tags]
               [-B suffix] [-l linkflags] [-N dbdir]
               file1 ... fileN directory
       install -dU [-vU] [-g group] [-m mode] [-N dbdir] [-o owner]
               [-M log] [-D dest] [-h hash] [-T tags]
               directory ...
zsh: command not found: LOAD
(stockmarket) moose@m3 yahoo-stock % curl install.duckdb.org | sh
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  3301  100  3301    0     0   8213      0 --:--:-- --:--:-- --:--:--  8211

*** DuckDB Linux/MacOS installation script, version 1.2.0 ***


         .;odxdl,            
       .xXXXXXXXXKc          
       0XXXXXXXXXXXd  cooo:  
      ,XXXXXXXXXXXXK  OXXXXd 
       0XXXXXXXXXXXo  cooo:  
       .xXXXXXXXXKc          
         .;odxdl,  


######################################################################## 100.0%

Successfully installed DuckDB binary to /Users/moose/.duckdb/cli/1.2.0/duckdb
  with a link from                      /Users/moose/.duckdb/cli/latest/duckdb

Hint: Append the following line to your shell profile:
export PATH='/Users/moose/.duckdb/cli/latest':$PATH


To launch DuckDB now, type
/Users/moose/.duckdb/cli/latest/duckdb
(stockmarket) moose@m3 yahoo-stock % /Users/moose/.duckdb/cli/latest/duckdb
v1.2.0 5f5512b827
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database.
D help
  ls
  help
  ".help"
  ls
  ".open stockdb
  ls
  help
  syntax off
  exit
  
D ls
‣ 
zsh: suspended  /Users/moose/.duckdb/cli/latest/duckdb
(stockmarket) moose@m3 yahoo-stock % cli
zsh: command not found: cli
(stockmarket) moose@m3 yahoo-stock % df
Filesystem     512-blocks      Used  Available Capacity iused      ifree %iused  Mounted on
/dev/disk3s1s1 1942700360  21786800 1489724144     2%  411592 4292601683    0%   /
devfs                 403       403          0   100%     700          0  100%   /dev
/dev/disk3s6   1942700360        40 1489724144     1%       0 7448620720    0%   /System/Volumes/VM
/dev/disk3s2   1942700360  13641728 1489724144     1%    1651 7448620720    0%   /System/Volumes/Preboot
/dev/disk3s4   1942700360      7272 1489724144     1%      53 7448620720    0%   /System/Volumes/Update
/dev/disk1s2      1024000     12328     985152     2%       1    4925760    0%   /System/Volumes/xarts
/dev/disk1s1      1024000     11016     985152     2%      34    4925760    0%   /System/Volumes/iSCPreboot
/dev/disk1s3      1024000      5784     985152     1%      87    4925760    0%   /System/Volumes/Hardware
/dev/disk3s5   1942700360 415116904 1489724144    22% 3038253 7448620720    0%   /System/Volumes/Data
map auto_home           0         0          0   100%       0          0     -   /System/Volumes/Data/home
(stockmarket) moose@m3 yahoo-stock % df()
function> arrow()
function function> 
(stockmarket) moose@m3 yahoo-stock % 
(stockmarket) moose@m3 yahoo-stock % pl()
function> 
(stockmarket) moose@m3 yahoo-stock % curl install.duckdb.org | sh
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  3301  100  3301    0     0   2133      0  0:00:01  0:00:01 --:--:--  2133

*** DuckDB Linux/MacOS installation script, version 1.2.0 ***


         .;odxdl,            
       .xXXXXXXXXKc          
       0XXXXXXXXXXXd  cooo:  
      ,XXXXXXXXXXXXK  OXXXXd 
       0XXXXXXXXXXXo  cooo:  
       .xXXXXXXXXKc          
         .;odxdl,  


Destination binary /Users/moose/.duckdb/cli/1.2.0/duckdb already exists

Hint: Append the following line to your shell profile:
export PATH='/Users/moose/.duckdb/cli/latest':$PATH


To launch DuckDB now, type
/Users/moose/.duckdb/cli/latest/duckdb
(stockmarket) moose@m3 yahoo-stock % df = con.execute("SELECT * FROM items").fetchdf()
print(df)
zsh: no matches found: con.execute(SELECT * FROM items).fetchdf
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-news.py		yahoo1-nyse.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stocks.db		yahoo-rss.py		yahoo1.py
__pycache__		duckdb_data		forexduckdb-query.py	yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	yahoo-feedparser.py	yahoo.py
(stockmarket) moose@m3 yahoo-stock % cd duckdb_data
(stockmarket) moose@m3 duckdb_data % df = con.execute("SELECT * FROM items").fetchdf()
print(df)
zsh: no matches found: con.execute(SELECT * FROM items).fetchdf
(stockmarket) moose@m3 duckdb_data % arr = con.execute("SELECT * FROM items").fetchnumpy()
print(arr)
zsh: no matches found: con.execute(SELECT * FROM items).fetchnumpy
(stockmarket) moose@m3 duckdb_data % arr = con.execute("SELECT * FROM items").fetchnumpy()
print(arr)
zsh: no matches found: con.execute(SELECT * FROM items).fetchnumpy
(stockmarket) moose@m3 duckdb_data % datetime.datetime
zsh: command not found: datetime.datetime
(stockmarket) moose@m3 duckdb_data % LIST
zsh: command not found: LIST
(stockmarket) moose@m3 duckdb_data % numpy.ndarray
zsh: command not found: numpy.ndarray
(stockmarket) moose@m3 duckdb_data % numpy.datetime64
zsh: command not found: numpy.datetime64
(stockmarket) moose@m3 duckdb_data % fetchnumpy()
function> df()
function function> fetch_df()
function function function> fetch_arrow_table()
function function function function> 
(stockmarket) moose@m3 duckdb_data % ls
Test.db			default.db		forex-duckdb.db		nasdaq_stocks.db	stocks.db
(stockmarket) moose@m3 duckdb_data % cd ..
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-news.py		yahoo1-nyse.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stocks.db		yahoo-rss.py		yahoo1.py
__pycache__		duckdb_data		forexduckdb-query.py	yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	yahoo-feedparser.py	yahoo.py
(stockmarket) moose@m3 yahoo-stock % python forexduckdb-query.py
(stockmarket) moose@m3 yahoo-stock % python forexduckdb-query.py

Query Results:
        date    pair      open      high       low     close  adj_close  volume
0 2024-02-23  EURUSD  1.082567  1.083940  1.081373  1.082567   1.082567     0.0
1 2024-02-26  EURUSD  1.081958  1.085906  1.081338  1.082005   1.082005     0.0
2 2024-02-27  EURUSD  1.085093  1.086614  1.083365  1.085093   1.085093     0.0
3 2024-02-28  EURUSD  1.084481  1.084716  1.079867  1.084481   1.084481     0.0
4 2024-02-29  EURUSD  1.083882  1.085588  1.080567  1.083882   1.083882     0.0

Total rows in result: 252
(stockmarket) moose@m3 yahoo-stock % python forexduckdb-query.py

Query Results:
        date    pair      open      high       low     close  adj_close  volume
0 2024-02-23  EURUSD  1.082567  1.083940  1.081373  1.082567   1.082567     0.0
1 2024-02-26  EURUSD  1.081958  1.085906  1.081338  1.082005   1.082005     0.0
2 2024-02-27  EURUSD  1.085093  1.086614  1.083365  1.085093   1.085093     0.0
3 2024-02-28  EURUSD  1.084481  1.084716  1.079867  1.084481   1.084481     0.0
4 2024-02-29  EURUSD  1.083882  1.085588  1.080567  1.083882   1.083882     0.0

Total rows in result: 252
(stockmarket) moose@m3 yahoo-stock % python forexduckdb-query.py

Query Results:
        date    pair       open       high        low      close  adj_close  volume
0 2024-02-23  AUDJPY  98.699997  99.045998  98.602997  98.699997  98.699997     0.0
1 2024-02-26  AUDJPY  98.776001  98.776001  98.466003  98.799004  98.799004     0.0
2 2024-02-27  AUDJPY  98.455002  98.611000  98.208000  98.443001  98.443001     0.0
3 2024-02-28  AUDJPY  98.473999  98.531998  97.809998  98.473999  98.473999     0.0
4 2024-02-29  AUDJPY  97.837997  97.837997  97.345001  97.820999  97.820999     0.0

Total rows in result: 5040
(stockmarket) moose@m3 yahoo-stock % python forexduckdb-query.py

Available Currency Pairs:
- AUDJPY
- AUDUSD
- CADJPY
- CHFJPY
- EURAUD
- EURCAD
- EURCHF
- EURGBP
- EURJPY
- EURUSD
- GBPAUD
- GBPCAD
- GBPCHF
- GBPJPY
- GBPUSD
- NZDJPY
- NZDUSD
- USDCAD
- USDCHF
- USDJPY

Total number of pairs: 20
(stockmarket) moose@m3 yahoo-stock % python forexduckdb-query.py

Available Columns:
- date
- pair
- open
- high
- low
- close
- adj_close
- volume

Sample Data (First 10 rows):
        date    pair       open       high        low      close  adj_close  volume
0 2024-02-23  AUDJPY  98.699997  99.045998  98.602997  98.699997  98.699997     0.0
1 2024-02-26  AUDJPY  98.776001  98.776001  98.466003  98.799004  98.799004     0.0
2 2024-02-27  AUDJPY  98.455002  98.611000  98.208000  98.443001  98.443001     0.0
3 2024-02-28  AUDJPY  98.473999  98.531998  97.809998  98.473999  98.473999     0.0
4 2024-02-29  AUDJPY  97.837997  97.837997  97.345001  97.820999  97.820999     0.0
5 2024-03-01  AUDJPY  97.531998  98.077003  97.545998  97.531998  97.531998     0.0
6 2024-03-04  AUDJPY  97.980003  98.139000  97.873001  97.980003  97.980003     0.0
7 2024-03-05  AUDJPY  97.905998  97.952003  97.438004  97.905998  97.905998     0.0
8 2024-03-06  AUDJPY  97.575996  98.205002  97.387001  97.575996  97.575996     0.0
9 2024-03-07  AUDJPY  98.007004  98.029999  97.432999  98.007004  98.007004     0.0

Total rows in dataset: 10
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-feedparser.py	yahoo.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stockduckdb.py		yahoo-news.py		yahoo1-nyse.py
__pycache__		duckdb_data		forexduckdb-query.py	stocks.db		yahoo-rss.py		yahoo1.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
(stockmarket) moose@m3 yahoo-stock % ls
Cboedata.py		data.py			forex-plot.py		stock.db		yahoo-data-news.py	yahoo-zmq.py		zmq-yahoo-client.py
Nasqdata.py		download_stats_a.txt	forexdata.py		stockduckdb-query.py	yahoo-feedparser.py	yahoo.py
__pycache__		duckdb_data		forexduckdb-query.py	stockduckdb.py		yahoo-news.py		yahoo1-nyse.py
data-yahoo-finance.py	forex-duckdb.db		nasdaq_stocks.db	stocks.db		yahoo-rss.py		yahoo1.py
(stockmarket) moose@m3 yahoo-stock % python stockduckdb-query.py
Traceback (most recent call last):
  File "/Users/moose/stock-api/yahoo-stock/stockduckdb-query.py", line 6, in <module>
    columns = con.execute("""
duckdb.duckdb.CatalogException: Catalog Error: Table with name stocks_prices does not exist!
Did you mean "duckdb_types"?
LINE 3:     FROM stocks_prices 
                 ^
(stockmarket) moose@m3 yahoo-stock % python stockduckdb-query.py
Traceback (most recent call last):
  File "/Users/moose/stock-api/yahoo-stock/stockduckdb-query.py", line 6, in <module>
    columns = con.execute("""
duckdb.duckdb.CatalogException: Catalog Error: Table with name ticker_prices does not exist!
Did you mean "duckdb_types"?
LINE 3:     FROM ticker_prices 
                 ^
(stockmarket) moose@m3 yahoo-stock % python stockduckdb-query.py
Traceback (most recent call last):
  File "/Users/moose/stock-api/yahoo-stock/stockduckdb-query.py", line 6, in <module>
    columns = con.execute("""
duckdb.duckdb.CatalogException: Catalog Error: Table with name ticker_prices does not exist!
Did you mean "duckdb_types"?
LINE 3:     FROM ticker_prices 
                 ^
(stockmarket) moose@m3 yahoo-stock % python stockduckdb-query.py

Available Tables:
Empty DataFrame
Columns: [table_name]
Index: []
(stockmarket) moose@m3 yahoo-stock % python stockduckdb-query.py

Database files in current directory:
- forex-duckdb.db
- s.db
- stocks.db
- nasdaq_stocks.db
- stock.db

Attempting to connect to: stock-duckdb.db

Available Tables:
Empty DataFrame
Columns: [table_name]
Index: []
(stockmarket) moose@m3 yahoo-stock % python stockduckdb-query.py

Database files in current directory:
- stock-duckdb.db
- forex-duckdb.db
- s.db
- stocks.db
- nasdaq_stocks.db
- stock.db

Attempting to connect to: stocks.db

Available Tables:
     table_name
0  stock_prices

Table Structure:

Structure for table: stock_prices
  column_name data_type
0        date      DATE
1      ticker   VARCHAR
2        open    DOUBLE
3        high    DOUBLE
4         low    DOUBLE
5       close    DOUBLE
6   adj_close    DOUBLE
7      volume    BIGINT
(stockmarket) moose@m3 yahoo-stock % 
