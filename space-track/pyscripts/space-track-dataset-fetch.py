import configparser
from datetime import datetime, timedelta

import requests
import os

# Define download function
def download_csv(session, start_date, end_date):
    url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/MEAN_MOTION/%3E11.25/EPOCH/%3E{start_date}%2C%3C{end_date}/ECCENTRICITY/%3C0.25/OBJECT_TYPE/PAYLOAD%2CDEBRIS%2CROCKET%20BODY/orderby/EPOCH%2CNORAD_CAT_ID%20asc/format/csv/"
    filename = f"../datasets/space-track_{start_date}.csv"
    response = session.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

def main():
    # Space-Track credentials
    # ACTION REQUIRED FOR YOU:
    # =========================
    # Provide a config file in the same directory as this file, called SLTrack.ini, with this format (without the # signs)
    # [configuration]
    # username = XXX
    # password = YYY
    # output = ZZZ
    #
    # ... where XXX and YYY are your www.space-track.org credentials (https://www.space-track.org/auth/createAccount for free account)
    # ... and ZZZ is your Excel Output file - e.g. starlink-track.xlsx (note: make it an .xlsx file)

    # Use configparser package to pull in the ini file (pip install configparser)
    config = configparser.ConfigParser()
    config.read("./SLTrack.ini")
    configUsr = config.get("configuration", "username")
    configPwd = config.get("configuration", "password")

    login_data = {'identity': configUsr, 'password': configPwd}



    # Define list of dates
    today = datetime(2024, 1, 29).date()
    days_prior = 33
    one_month_ago = today - timedelta(days=days_prior)
    dates = [d.strftime("%Y-%m-%d") for d in (today - timedelta(n) for n in range((today - one_month_ago).days))]
    dates.sort()

    # Login to Space-Track
    login_url = "https://www.space-track.org/ajaxauth/login"
    session = requests.Session()
    response = session.post(login_url, data=login_data)

    # Check if login successful
    if response.status_code != 200:
        print("Login failed. Please check your credentials.")
        exit()

    # Iterate through dates and download CSV files
    for i in range(len(dates) - 1):
        start_date = dates[i]
        end_date = dates[i + 1]
        download_csv(session, start_date, end_date)

    # Logout from Space-Track
    logout_url = "https://www.space-track.org/ajaxauth/logout"
    session.get(logout_url)

    print("All files downloaded and logged out successfully.")

if __name__ == "__main__":
    main()