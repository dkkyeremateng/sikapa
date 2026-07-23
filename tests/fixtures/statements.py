SAMPLE_STATEMENT = '''Statement,Header,Field Name,Field Value
Statement,Data,BrokerName,Interactive Brokers LLC
Statement,Data,Period,"April 1, 2026 - May 19, 2026"
Trades,Header,DataDiscriminator,Asset Category,Currency,Account,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code
Trades,Data,Order,Stocks,USD,U1111111,AMZN,"2026-04-01, 12:14:23",5.25,200,198.5,-1050,-0.35,1050.35,0,-7.875,FPA;O;P
Trades,SubTotal,,Stocks,USD,AMZN,,,5.25,,,-1050,-0.35,1050.35,0,-7.875,
Trades,Data,Order,Stocks,USD,U1111111,MSFT,"2026-04-01, 12:12:16",1.5,380,375,-570,-0.35,570.35,0,-7.5,O;RP
Trades,SubTotal,,Stocks,USD,MSFT,,,1.5,,,-570,-0.35,570.35,0,-7.5,
Trades,Total,,Stocks,USD,,,,,,,-1620,-0.7,1620.7,0,-15.375,
Deposits & Withdrawals,Header,Currency,Account,Settle Date,Description,Amount
Deposits & Withdrawals,Data,USD,U1111111,2026-04-30,Electronic Fund Transfer,1500
Deposits & Withdrawals,Data,Total,,,,1500
Dividends,Header,Currency,Account,Date,Description,Amount
Dividends,Data,USD,U1111111,2026-04-01,NVDA(US67066G1040) Cash Dividend USD 0.01 per Share (Ordinary Dividend),0.2
Dividends,Data,USD,U1111111,2026-04-08,NVO(US6701002056) Cash Dividend USD 1.25 per Share (Ordinary Dividend),50
Dividends,Data,Total,,,,50.2
Withholding Tax,Header,Currency,Account,Date,Description,Amount,Code
Withholding Tax,Data,USD,U1111111,2026-04-08,NVO(US6701002056) Cash Dividend USD 1.25 per Share - DK Tax,-13.5,
Withholding Tax,Data,Total,,,,-13.5,
Fees,Header,Subtitle,Currency,Account,Date,Description,Amount
Fees,Data,Other Fees,USD,U1111111,2026-04-08,NVO(US6701002056) Cash Dividend USD 1.25 per Share - FEE,-0.6
Fees,Data,Total,,,,,-0.6
Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code
Open Positions,Data,Summary,Stocks,USD,AMZN,12,1,180,2160,230,2760,600,
Open Positions,Data,Summary,Stocks,USD,MSFT,3,1,390,1170,420,1260,90,
Open Positions,Total,,Stocks,USD,,,,,3330,,4020,690,
Financial Instrument Information,Header,Asset Category,Symbol,Description,Conid,Security ID,Underlying,Listing Exch,Multiplier,Type,Code
Financial Instrument Information,Data,Stocks,AMZN,AMAZON.COM INC,3691937,US0231351067,AMZN,NASDAQ,1,COMMON,
Financial Instrument Information,Data,Stocks,MSFT,MICROSOFT CORP,272093,US5949181045,MSFT,NASDAQ,1,COMMON,
Net Asset Value,Header,Asset Class,Prior Total,Current Long,Current Short,Current Total,Change
Net Asset Value,Data,Cash ,1200,800,0,800,-400
Net Asset Value,Data,Stock,20000,24000,0,24000,4000
Net Asset Value,Data,Total,21200,24800,0,24800,3600
Net Asset Value,Header,Time Weighted Rate of Return
Net Asset Value,Data,12.5%
Corporate Actions,Header,Asset Category,Currency,Account,Report Date,Date/Time,Description,Quantity,Proceeds,Value,Realized P/L,Code
Corporate Actions,Data,Stocks,USD,U1111111,2026-04-11,"2026-04-10, 20:25:00","SCHD(US8085247976) Split 3 for 1 (SCHD, SCHWAB US DVD EQUITY ETF, US8085247976)",30,0,0,0,
Corporate Actions,Data,Total,,,,,,,0,0,0,
'''


def mini_statement(account, period, twrr, nav_current, currency="USD", div=10.0):
    """Minimal statement: one dividend row (sets the account), a NAV row, TWRR."""
    return (
        f'Statement,Data,Period,"{period}"\n'
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        f"Dividends,Data,{currency},{account},2024-06-01,ACME Cash Dividend,{div}\n"
        "Net Asset Value,Header,Asset Class,Prior Total,Current Long,Current Short,Current Total,Change\n"
        f"Net Asset Value,Data,Stock,0,{nav_current},0,{nav_current},{nav_current}\n"
        "Net Asset Value,Header,Time Weighted Rate of Return\n"
        f"Net Asset Value,Data,{twrr}\n"
    )


BUYSELL_STATEMENT = (
    "Statement,Data,Period,\"January 1, 2024 - December 31, 2025\"\n"
    "Trades,Header,DataDiscriminator,Asset Category,Currency,Account,Symbol,Date/Time,Quantity,T. Price,C. Price,Proceeds,Comm/Fee,Basis,Realized P/L,MTM P/L,Code\n"
    'Trades,Data,Order,Stocks,USD,U1,AMZN,"2024-01-01, 10:00:00",10,100,100,-1000,0,1000,0,0,\n'
    'Trades,Data,Order,Stocks,USD,U1,AMZN,"2025-02-05, 10:00:00",-6,150,150,900,0,-600,300,0,\n'
    'Trades,Data,Order,Stocks,USD,U1,MSFT,"2025-03-01, 10:00:00",4,50,50,-200,0,200,0,0,\n'
    'Trades,Data,Order,Stocks,USD,U1,MSFT,"2025-03-31, 10:00:00",-4,40,40,160,0,-160,-40,0,\n'
    "Dividends,Header,Currency,Account,Date,Description,Amount\n"
    "Dividends,Data,USD,U1,2025-04-01,AMZN(US1) Cash Dividend,12\n"
    "Dividends,Data,EUR,U1,2025-04-02,SAP(DE1) Cash Dividend,8\n"
    "Withholding Tax,Header,Currency,Account,Date,Description,Amount,Code\n"
    "Withholding Tax,Data,USD,U1,2025-04-01,AMZN(US1) Cash Dividend - US Tax,-2,\n"
    "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code\n"
    "Open Positions,Data,Summary,Stocks,USD,AMZN,4,1,100,400,160,640,240,\n"
    "Open Positions,Data,Summary,ETFs,USD,VOO,10,1,50,500,60,600,100,\n"
)


# A minimal but complete OFX 2.x (XML) *investment* statement, for the
# cross-broker OFX/QFX importer. Exercises the security master (SECLIST maps a
# CUSIP to the AMZN ticker), a BUYSTOCK and a SELLSTOCK (sign normalization), an
# INCOME dividend, an INVBANKTRAN deposit, and an INVPOSLIST position.
SAMPLE_OFX = (
    '<?xml version="1.0" encoding="US-ASCII"?>\n'
    '<?OFX OFXHEADER="200" VERSION="220" SECURITY="NONE" OLDFILEUID="NONE" NEWFILEUID="NONE"?>\n'
    "<OFX>"
    "<SIGNONMSGSRSV1><SONRS><STATUS><CODE>0</CODE><SEVERITY>INFO</SEVERITY></STATUS>"
    "<DTSERVER>20260515120000</DTSERVER><LANGUAGE>ENG</LANGUAGE></SONRS></SIGNONMSGSRSV1>"
    "<INVSTMTMSGSRSV1><INVSTMTTRNRS><TRNUID>1</TRNUID>"
    "<STATUS><CODE>0</CODE><SEVERITY>INFO</SEVERITY></STATUS>"
    "<INVSTMTRS><DTASOF>20260515120000</DTASOF><CURDEF>USD</CURDEF>"
    "<INVACCTFROM><BROKERID>vanguard.com</BROKERID><ACCTID>1234567</ACCTID></INVACCTFROM>"
    "<INVTRANLIST><DTSTART>20260401000000</DTSTART><DTEND>20260515000000</DTEND>"
    "<BUYSTOCK><INVBUY><INVTRAN><FITID>T1</FITID><DTTRADE>20260402000000</DTTRADE>"
    "<DTSETTLE>20260404000000</DTSETTLE><MEMO>Buy AMZN</MEMO></INVTRAN>"
    "<SECID><UNIQUEID>023135106</UNIQUEID><UNIQUEIDTYPE>CUSIP</UNIQUEIDTYPE></SECID>"
    "<UNITS>10</UNITS><UNITPRICE>180.00</UNITPRICE><COMMISSION>1.00</COMMISSION>"
    "<FEES>0.05</FEES><TOTAL>-1801.05</TOTAL><SUBACCTSEC>CASH</SUBACCTSEC>"
    "<SUBACCTFUND>CASH</SUBACCTFUND></INVBUY><BUYTYPE>BUY</BUYTYPE></BUYSTOCK>"
    "<SELLSTOCK><INVSELL><INVTRAN><FITID>T2</FITID><DTTRADE>20260420000000</DTTRADE>"
    "<MEMO>Sell AMZN</MEMO></INVTRAN>"
    "<SECID><UNIQUEID>023135106</UNIQUEID><UNIQUEIDTYPE>CUSIP</UNIQUEIDTYPE></SECID>"
    "<UNITS>-4</UNITS><UNITPRICE>190.00</UNITPRICE><COMMISSION>1.00</COMMISSION>"
    "<TOTAL>759.00</TOTAL><SUBACCTSEC>CASH</SUBACCTSEC>"
    "<SUBACCTFUND>CASH</SUBACCTFUND></INVSELL><SELLTYPE>SELL</SELLTYPE></SELLSTOCK>"
    "<INCOME><INVTRAN><FITID>D1</FITID><DTTRADE>20260410000000</DTTRADE>"
    "<MEMO>AMZN DIVIDEND</MEMO></INVTRAN>"
    "<SECID><UNIQUEID>023135106</UNIQUEID><UNIQUEIDTYPE>CUSIP</UNIQUEIDTYPE></SECID>"
    "<INCOMETYPE>DIV</INCOMETYPE><TOTAL>5.50</TOTAL><SUBACCTSEC>CASH</SUBACCTSEC>"
    "<SUBACCTFUND>CASH</SUBACCTFUND></INCOME>"
    "<INVBANKTRAN><STMTTRN><TRNTYPE>DEP</TRNTYPE><DTPOSTED>20260405000000</DTPOSTED>"
    "<TRNAMT>1000.00</TRNAMT><FITID>B1</FITID><NAME>ACH DEPOSIT</NAME></STMTTRN>"
    "<SUBACCTFUND>CASH</SUBACCTFUND></INVBANKTRAN>"
    "</INVTRANLIST>"
    "<INVPOSLIST><POSSTOCK><INVPOS>"
    "<SECID><UNIQUEID>023135106</UNIQUEID><UNIQUEIDTYPE>CUSIP</UNIQUEIDTYPE></SECID>"
    "<HELDINACCT>CASH</HELDINACCT><POSTYPE>LONG</POSTYPE><UNITS>6</UNITS>"
    "<UNITPRICE>185.00</UNITPRICE><MKTVAL>1110.00</MKTVAL>"
    "<DTPRICEASOF>20260515000000</DTPRICEASOF></INVPOS></POSSTOCK></INVPOSLIST>"
    "</INVSTMTRS></INVSTMTTRNRS></INVSTMTMSGSRSV1>"
    "<SECLISTMSGSRSV1><SECLIST><STOCKINFO><SECINFO>"
    "<SECID><UNIQUEID>023135106</UNIQUEID><UNIQUEIDTYPE>CUSIP</UNIQUEIDTYPE></SECID>"
    "<SECNAME>AMAZON COM INC</SECNAME><TICKER>AMZN</TICKER></SECINFO></STOCKINFO>"
    "</SECLIST></SECLISTMSGSRSV1>"
    "</OFX>"
)
