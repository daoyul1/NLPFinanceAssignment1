#!/un3 sr/bin/Python
"""
    Program to download EDGAR files by form type
    ND-SRAF / McDonald : 201606
    https://sraf.nd.edu
    Dependencies (i.e., modules you must already have downloaded)
      EDGAR_Forms.py
      EDGAR_Pac.py
      General_Utilities.py
"""
import pandas as pd
import os
import re
import time
import sys
# Modify the following statement to identify the path for local modules
# sys.path.append('D:\GD\Python\TextualAnalysis\Modules')
# Since these imports are dynamically mapped your IDE might flag an error...it's OK
import EDGAR_Forms  # This module contains some predefined form groups
import EDGAR_Pac
import General_Utilities



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * +

#  NOTES
#        The EDGAR archive contains millions of forms.
#        For details on accessing the EDGAR servers see:
#          https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
#        From that site:
#            "To preserve equitable server access, we ask that bulk FTP
#             transfer requests be performed between 9 PM and 6 AM Eastern 
#             time. Please use efficient scripting, downloading only what you
#             need and space out requests to minimize server load."
#        Note that the program will check the clock every 10 minutes and only
#            download files during the appropriate time.
#        Be a good citizen...keep your requests targeted.
#
#        For large downloads you will sometimes get a hiccup in the server
#            and the file request will fail.  These errs are documented in
#            the log file.  You can manually download those files that fail.
#            Although I attempt to work around server errors, if the SEC's server
#            is sufficiently busy, you might have to try another day.
#
#       For a list of form types and counts by year:
#         "All SEC EDGAR Filings by Type and Year"
#          at https://sraf.nd.edu/textual-analysis/resources/#All%20SEC%20EDGAR%20Filings%20by%20Type%20and%20Year


# -----------------------
# User defined parameters
# -----------------------

# List target forms as strings separated by commas (case sensitive) or
#   load from EDGAR_Forms.  (See EDGAR_Forms module for predefined lists.)
PARM_FORMS = EDGAR_Forms.f_10X  # All 10-K/10-Q variants or, for example, PARM_FORMS = ['8-K', '8-K/A']
#PARM_FORMS = ['10-K'] # ONLY need 10-K report
PARM_BGNYEAR = 2020  # User selected bgn period.  Earliest available is 1994
PARM_ENDYEAR = 2024  # User selected end period.
PARM_BGNQTR = 1  # Beginning quarter of each year
PARM_ENDQTR = 4  # Ending quarter of each year
# Path where you will store the downloaded files
PARM_PATH = r'D:\data'
# Change the file pointer below to reflect your location for the log file
#    (directory must already exist)
PARM_LOGFILE = (r'C:\Users\lidao\PycharmProjects\pythonProject4\result' +
                r'EDGAR_Download_FORM-X_LogFile_' +
                str(PARM_BGNYEAR) + '-' + str(PARM_ENDYEAR) + '.txt')
# EDGAR parameter
PARM_EDGARPREFIX = 'https://www.sec.gov/Archives/'


#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * +
# only need the dow30 stocks
# dows=pd.read_csv("dow30.csv",header=None,index_col=0)
# ciks=pd.read_csv("cik.csv",header=None,index_col=0)
# ciks.index=[i.upper() for i in ciks.index]
# dow_cik=list(map(int,pd.concat([ciks,dows],axis=1,join="inner",sort=True).iloc[:,0].tolist()))

# def dow30filter(all_index):
#     filtered=[]
#     for idx in all_index:
#         if idx.cik in dow_cik:
#             filtered.append(idx)
#     return filtered

# SP500 Part
SP500_CIK_DF = pd.read_csv("sp500_cik.csv", dtype={'cik': int})
SP500_CIK_DF['quarter'] = SP500_CIK_DF['quarter'].str.upper()


def sp500_filter(all_index, year, qtr):
    """Filter filings based on quarter-specific CIK mapping"""
    filtered = []
    target_quarter = f"{year}Q{qtr}"

    # Get valid CIKs for this quarter
    valid_ciks = SP500_CIK_DF.loc[
        SP500_CIK_DF['quarter'] == target_quarter, 'cik'
    ].unique().tolist()

    if not valid_ciks:
        return filtered

    # Split into 10-Q and 10-K groups
    tenq_items = []
    tenk_items = []

    for item in all_index:
        if item.cik not in valid_ciks:
            continue

        if item.form in EDGAR_Forms.f_10Q + EDGAR_Forms.f_10QA + EDGAR_Forms.f_10QT:
            tenq_items.append(item)
        elif item.form in EDGAR_Forms.f_10K + EDGAR_Forms.f_10KA + EDGAR_Forms.f_10KT:
            tenk_items.append(item)

    # Process 10-Qs first
    processed_ciks = set()
    for item in tenq_items:
        filtered.append(item)
        processed_ciks.add(item.cik)

    # Add 10-Ks for CIKs without 10-Q
    for item in tenk_items:
        if item.cik not in processed_ciks:
            filtered.append(item)

    return filtered

def download_forms():
    # Download each year/quarter master.idx and save record for requested forms
    f_log = open(PARM_LOGFILE, 'a')
    f_log.write('BEGIN LOOPS:  {0}\n'.format(time.strftime('%c')))
    n_tot = 0
    n_errs = 0
    for year in range(PARM_BGNYEAR, PARM_ENDYEAR + 1):
        for qtr in range(PARM_BGNQTR, PARM_ENDQTR + 1):
            startloop = time.perf_counter()  # Replaced time.clock()
            n_qtr = 0
            file_count = {}
            # Setup output path
            path = os.path.join(PARM_PATH, str(year), f'QTR{qtr}')
            if not os.path.exists(path):
                os.makedirs(path)
                print('Path: {0} created'.format(path))
            master_index = EDGAR_Pac.download_masterindex(year, qtr, True)
            masterindex = sp500_filter(master_index, year, qtr)  # Change to SP500 filter
            if masterindex:
                for item in masterindex:
                    # while EDGAR_Pac.edgar_server_not_available(True):  # kill time when server not available
                    #     pass
                    if item.form in PARM_FORMS:
                        n_qtr += 1
                        # Keep track of filings and identify duplicates
                        fid = str(item.cik) + str(item.filingdate) + item.form
                        if fid in file_count:
                            file_count[fid] += 1
                        else:
                            file_count[fid] = 1
                        # Setup EDGAR URL and output file name
                        url = PARM_EDGARPREFIX + item.path
                        fname = (os.path.join(path, f"{item.filingdate}_{item.form.replace('/', '-')}_"
                                   f"{item.path.replace('/', '_').replace('.txt', '')}"
                                   f"_{file_count[fid]}.txt"))
                        return_url = General_Utilities.download_to_file(url, fname, f_log)
                        if return_url:
                            n_errs += 1
                        n_tot += 1
                        # time.sleep(1)  # Space out requests
            print(f"{year}:{qtr} -> {n_qtr:,} downloads completed.  Time = "
                  f"{time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - startloop))} | "
                  f"{time.strftime('%c')}")
            f_log.write(f"{year} | {qtr} | n_qtr = {n_qtr:>8,} | n_tot = {n_tot:>8,} | "
                        f"n_err = {n_errs:>6,} | {time.strftime('%c')}\n")
            f_log.flush()

    print(f'{n_tot:,} total forms downloaded.')
    f_log.write(f'\n{n_tot:,} total forms downloaded.')

if __name__ == '__main__':
    start = time.perf_counter()  # Replaced time.clock()
    print('\n' + time.strftime('%c') + '\nND_SRAF:  Program EDGAR_DownloadForms.py\n')
    download_forms()
    print('\nEDGAR_DownloadForms.py | Normal termination | ' +
          time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start)))  # Replaced time.clock()
    print(time.strftime('%c'))
