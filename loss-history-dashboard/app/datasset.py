import numpy as np
import pandas as pd

def create_full_dummy_data(n_samples=1000, output_file='dummy_full_loss_history.csv'):
    np.random.seed(42)
    
    # Core fields
    df = pd.DataFrame({
        'pn': np.arange(1000001, 1000001 + n_samples),  # dummy policy number
        'mod': np.random.randint(0, 6, size=n_samples),
        'eey': np.round(np.random.uniform(0.1, 1.0, size=n_samples), 3),
        'cc_total': np.random.poisson(1.0, size=n_samples),
    })
    df['il_total'] = np.round(df['cc_total'] * np.random.uniform(100, 1000, size=n_samples), 2)
    df['trgt'] = np.round(df['il_total'] / df['eey'], 2)
    df['wt']   = df['eey']
    
    # Raw history attributes (48): 3 classes * 16 perils
    raw_prefixes = ['num_loss_3yr_', 'num_loss_yrs45_', 'loss_free_yrs_']
    raw_perils = [
        'fire', 'windhail', 'windhail_sm', 'windhail_lg', 'water', 'water_wthr',
        'water_othr', 'lightning', 'theft', 'other_w_snow', 'other_wo_snow',
        'snow', 'sewer', 'liability', 'nonw', 'total'
    ]
    for pref in raw_prefixes:
        for p in raw_perils:
            df[f'{pref}{p}'] = np.random.randint(0, 6, size=n_samples)
    
    # Weighted history attributes (56): 4 weighting schemes * 14 perils
    weight_prefixes = ['lhdwc_5y_1d_', 'lhdwc_5y_2d_', 'lhdwc_5y_3d_', 'lhdwc_5y_4d_']
    weight_perils = [
        'fire', 'wind', 'hail', 'water', 'water_nw', 'water_w', 'theft', 'other',
        'sewer', 'weather', 'nonweather', 'small', 'large', 'total'
    ]
    for pref in weight_prefixes:
        for p in weight_perils:
            df[f'{pref}{p}'] = np.round(np.random.uniform(0, 5, size=n_samples), 3)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dummy dataset with {df.shape[0]} rows and {df.shape[1]} columns saved to '{output_file}'.")

if __name__ == '__main__':
    create_full_dummy_data()
