import numpy as np
import QuantLib as ql

from libs.stochastic_processes import HestonProcess

 
class VanillaHestonQL(object):

    # Class attributes
    _dayCount = ql.SimpleDayCounter()
    _today = ql.Date.todaysDate()
    #_calendar = ql.NullCalendar()

    def __init__(self,
        flag: str,
        strike: float,
        maturity: float,  
        heston_process: HestonProcess
    ) -> None:
        # Defining option instance
        if flag == "c":
            CP = ql.Option.Call
        elif flag == "p":
            CP = ql.Option.Put
        else:
            raise ValueError("flag must be c for Call or p for Put!")

        self.option = ql.VanillaOption(
            payoff=ql.PlainVanillaPayoff(CP, strike),
            exercise=ql.EuropeanExercise(self._today + ql.Period(int(maturity * 360), ql.Days))
        )

        self.heston_process = heston_process

        # Defining non-changin model variables
        self.riskFreeRate = ql.FlatForward(self._today, self.heston_process.r, self._dayCount)
        self.dividendYield = ql.FlatForward(self._today, self.heston_process.q, self._dayCount)

    def evaluateArray(self, S_and_V_array: np.ndarray) -> np.ndarray:
        ql.Settings.instance().evaluationDate = self._today
        # Allocate sol-array
        uSol = np.zeros((S_and_V_array.shape[0],))
        # Loop over the underlying and variance
        for i, SV in enumerate(S_and_V_array):
            # Build Heston Model
            hestonProcess = ql.HestonProcess(
                ql.YieldTermStructureHandle(self.riskFreeRate),
                ql.YieldTermStructureHandle(self.dividendYield),
                ql.QuoteHandle(ql.SimpleQuote(SV[0])),
                SV[1], self.heston_process.kappa, self.heston_process.eta,
                self.heston_process.sigma, self.heston_process.rho
            )
            hestonModel = ql.HestonModel(hestonProcess)
            # Set engine
            engine = ql.AnalyticHestonEngine(hestonModel)
            self.option.setPricingEngine(engine)
            # Evaluate option 
            uSol[i] = self.option.NPV()

        return uSol
    
    def evaluate(self, SV: np.ndarray) -> np.ndarray:
        ql.Settings.instance().evaluationDate = self._today
        # Loop over the underlying and variance
        
        # Build Heston Model
        hestonProcess = ql.HestonProcess(
            ql.YieldTermStructureHandle(self.riskFreeRate),
            ql.YieldTermStructureHandle(self.dividendYield),
            ql.QuoteHandle(ql.SimpleQuote(SV[0])),
            SV[1], self.heston_process.kappa, self.heston_process.eta,
            self.heston_process.sigma, self.heston_process.rho
        )
        hestonModel = ql.HestonModel(hestonProcess)
        # Set engine
        engine = ql.AnalyticHestonEngine(hestonModel)
        self.option.setPricingEngine(engine)
        # Evaluate option 
        uSol = self.option.NPV()
        return uSol

    

