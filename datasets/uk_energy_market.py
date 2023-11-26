from typing import Dict
from .utils import load_pickle_from_url

LINK = 'https://github.com/clarkmaio/datasets/raw/main/uk_energy_market.pkl'


DESCR = '''
        UK energy market dataset
        -------------------------
        
        Description:
            The dataset contains several variable describing UK energy market.
            The main feature is da_price which represent day ahead gas prices.
            It is supposed to be the target of the regression problem you can approach using this dataset
        
        Columns:
            windspeed_forecast             float64
            solar_forecast                 float64
            wind_forecast                  float64
            embd_wind_forecast             float64
            battery_forecast               float64
            biomass_forecast               float64
            ccgt_forecast                  float64
            coal_forecast                  float64
            flexgen_forecast               float64
            hydro_forecast                 float64
            interconn_forecast             float64
            nuclear_forecast               float64
            oil_forecast                   float64
            pumpstorage_forecast           float64
            demand_system_margin           float64
            demand_forecast                float64
            within_day_availability        float64
            margin                         float64
            within_day_margin_forecast     float64
            longterm_wind                  float64
            longterm_wind_over_demand      float64
            longterm_wind_over_margin      float64
            longterm_solar_over_demand     float64
            longterm_solar_over_margin     float64
            margin_over_demand             float64
            snsp_forecast                  float64
            stack_price                    float64
            within_day_stack_price         float64
            margin_inertia_forecast        float64
            margin_battery_forecast        float64
            margin_biomass_forecast        float64
            margin_ccgt_forecast           float64
            margin_coal_forecast           float64
            margin_flexgen_forecast        float64
            margin_hydro_forecast          float64
            margin_oil_forecast            float64
            margin_pumpstorage_forecast    float64
            residual_demand                float64
            low_residual                     int32
            da_price_forecast              float64
            da_price                       float64
            dap_be_D-1                     float64
            dap_fr_D-1                     float64
            dap_gb_epex_D-1                float64
            dap_gb_nordpool_D-1            float64
            dap_isem_D-1                   float64
            dap_nl_D-1                     float64
            dap_no2_sdac_auction_D-1       float64
            dap_no2_nsl_auction_D-1        float64
            dap_be_D-7                     float64
            dap_fr_D-7                     float64
            dap_gb_epex_D-7                float64
            dap_gb_nordpool_D-7            float64
            dap_isem_D-7                   float64
            dap_nl_D-7                     float64
            dap_no2_sdac_auction_D-7       float64
            dap_no2_nsl_auction_D-7        float64
        
        Summary statistics:
            Too much variables to describe.
        '''

def load_uk_gas_price() -> Dict:
    df = load_pickle_from_url(LINK)
    df.rename(columns = {'target': 'da_price'}, inplace = True)

    dataset = {'data': df, 'DESCR': DESCR, 'columns': df.columns}
    return dataset


if __name__ == '__main__':
    dataset = load_uk_gas_price()
