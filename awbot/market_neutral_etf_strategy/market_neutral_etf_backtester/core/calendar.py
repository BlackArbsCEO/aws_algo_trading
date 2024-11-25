import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


class TradingCalendar(USFederalHolidayCalendar):
    def __init__(self):
        super().__init__()
        self.holidays = self.holidays()

    def is_business_day(self, date: pd.Timestamp) -> bool:
        """Check if date is a valid trading day"""
        date = pd.Timestamp(date).date()
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        return date not in self.holidays

    def next_business_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """Get next business day"""
        next_day = date + pd.Timedelta(days=1)
        while not self.is_business_day(next_day):
            next_day += pd.Timedelta(days=1)
        return next_day

    def first_business_day_of_next_month(self, date: pd.Timestamp) -> pd.Timestamp:
        """Get first business day of the next month"""
        # Move to first day of next month
        if date.month == 12:
            next_month = pd.Timestamp(year=date.year + 1, month=1, day=1)
        else:
            next_month = pd.Timestamp(year=date.year, month=date.month + 1, day=1)

        # Find first business day
        while not self.is_business_day(next_month):
            next_month += pd.Timedelta(days=1)

        return next_month
