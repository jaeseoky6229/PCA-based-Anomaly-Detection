import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mode_analysis

E_modulus = 2.1e11   # Young's Modulus (Pa)
I_moment = 8.33e-6   # Second moment of area (m^4)
rho_density = 7850   # Density (kg/m^3)
A_area = 0.02        # Cross-sectional area (m^2)

L_total_beam = 10.0  # Total length of the beam (m)
n_elements = 20
n_mode_get = 6
temp_variation = 'Quasi-linear'   # select 'Quasi-linear' or 'bi-linear'
damage_rate = 0.9
boundary_condition = 'simply-supported'  # 'cantilever', 'simply-supported'

# Run simulator based on the record of temperature
fn_temp_data = 'data/6시간단위_기후데이터.csv'
df_temp = pd.read_csv(fn_temp_data, encoding='cp949')
t = df_temp.iloc[:, -1].values
t_interval = 6  # time interval (hours)

# Setting damage information
damage_start_ind = t.shape[0] - 1141
damage_ele = [9, 10]
damage_severity = [damage_rate, damage_rate]

# Find natural frequencies
Data_generator = mode_analysis.Generate_dataset(
    n_elements=n_elements,
    temp_variation=temp_variation,
    E_modulus=E_modulus,
    I_moment=I_moment,
    rho_density=rho_density,
    A_area=A_area,
    L_total_beam=L_total_beam,
    plot_on=False
)

freq = []
freq_orign = []

for time_ind in range(t.shape[0]):
    if time_ind < damage_start_ind:
        Data_generator.set_parameter(
            t[time_ind],
            damage_ele=None,
            damage_severity=None
        )
    else:
        Data_generator.set_parameter(
            t[time_ind],
            damage_ele=damage_ele,
            damage_severity=damage_severity
        )

    frequencies_hz = Data_generator.run(boundary_condition=boundary_condition)
    freq.append(list(frequencies_hz[:n_mode_get]))

for time_ind in range(t.shape[0]):
    Data_generator.set_parameter(
        t[time_ind],
        damage_ele=None,
        damage_severity=None
    )
    frequencies_hz = Data_generator.run(boundary_condition=boundary_condition)
    freq_orign.append(list(frequencies_hz[:n_mode_get]))

df_result = pd.DataFrame(freq)
df_result = df_result.iloc[:400]

df_orign_result = pd.DataFrame(freq_orign)
df_orign_result = df_orign_result.iloc[:320]

for i in range(df_result.shape[1]):
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(df_result.iloc[:, i], label=f'mode{i+1}_defect')
    # plt.plot(df_orign_result.iloc[:, i], label=f'mode{i+1}_original')
    plt.axvline(x=320, color='r', linestyle='--', linewidth=2)
    plt.title(
        f"Frequency Variation Over Time ({boundary_condition})",
        fontsize=20,
        fontweight='bold'
    )
    plt.xlabel("sample(400th)", fontsize=20, fontweight='bold')
    plt.ylabel("Frequency (Hz)", fontsize=20, fontweight='bold')
    # plt.legend()
    plt.grid(True)
    plt.show()

df_orign_result.to_csv(
    f'data/df_origin_result({temp_variation},{boundary_condition},{t_interval}).csv',
    index=False,
    encoding='utf-8-sig'
)

df_result.to_csv(
    f'data/df_result({temp_variation},{damage_rate},{boundary_condition},{t_interval}).csv',
    index=False,
    encoding='utf-8-sig'
)