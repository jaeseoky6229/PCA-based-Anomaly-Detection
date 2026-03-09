import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt

class Generate_dataset():
	def __init__(self, n_elements, temp_variation = 'Quasi-linear',
			E_modulus = 2.1e11, I_moment = 8.33e-6, rho_density = 7850,
			A_area = 0.02, L_total_beam = 10.0, plot_on = True):

		self.n_elements = n_elements
		self.E_modulus_ = E_modulus
		self.I_moment_ = I_moment
		self.rho_density_ = rho_density
		self.A_area_ = A_area
		self.L_total_beam = L_total_beam
		self.L_element = self.L_total_beam / self.n_elements
		self.temp_variation = temp_variation
		
		# Temperature Variation		
		if self.temp_variation == 'Quasi-linear':
			# Define temperature versus Young's modulus
			T0 = np.arange(-20, 51, 5)
			Es = np.array([
					1.07523676300626, 1.06253244739916, 1.05077090900101, 
					1.03995214781181, 1.03007616383155, 1.02114295706025, 
					1.01315252749788, 1.00610487514447, 1, 
					0.994837902064479, 0.990618581337904, 0.987342037820276, 
					0.985008271511596, 0.983617282411862, 0.983169070521074
				])
			# Fit 2nd-order polynomials using polyfit
			self.polyfit = np.polyfit(T0, Es, 2)
			t = np.arange(-20, 51, 1)
			Factor = np.polyval(self.polyfit, t)

		elif self.temp_variation == 'bi-linear':
			#Define temperature versus Young's modulus
			T0_1 = np.array([-10, 0])
			Es_1 = np.array([1.2, 1])
			T0_2 = np.array([0, 30])
			Es_2 = np.array([1, 0.98])
			self.polyfit_1 = np.polyfit(T0_1, Es_1, 1)
			self.polyfit_2 = np.polyfit(T0_2, Es_2, 1)

			t = np.arange(-20, 51, 1)
			Factor = np.where(t<=0,
					 			np.polyval(self.polyfit_1, t), 
								np.polyval(self.polyfit_2, t))

		else :
			print("Please retry to Quasi-linear or bi-linear")
								
		if plot_on:
			plt.figure(figsize=(8,6), dpi=200)
			plt.plot(t, Factor, label=f'Fitted Curve ({temp_variation})')
			plt.axis('tight')
			font_options = {'fontsize': 20, 'fontweight': 'bold'}
			plt.title('Temperature Dependence of the Elastic Modulus', **font_options)
			plt.xlabel('Temperature(℃)', **font_options)
			plt.ylabel('Elastic Modulus Factor', **font_options)
			ax = plt.gca()
			ax.tick_params(axis='both', which='major', labelsize=12)
			# plt.legend()
			plt.grid(True)

	def set_parameter(self, t, damage_ele = None, damage_severity = None):
		"""
		Apply variation of Young's Modulus according to temperature and damage existence.
		
		Parameters:
		t (scalar): temperature values
		damage_ele(np.ndarray): indice for damaged elements (np.array([0, 10]), 1st and 11st elements)
		damage_severity(np.ndarray): percentage for damaged elements
		  => (np.array([0.8, 0.9]), 20% reduction for 1st element // 10% stiffness reduction for 11st element

		Returns:
		E_modulus (np.ndarray): Vector for Young's modulus (1-by-n_elem)
		I_moment (np.ndarray): Vector for I_moment (1-by-n_elem)
		rho_density (np.ndarray): Vector for rho_density (1-by-n_elem)
		A_area (np.ndarray): Vector for A_area (1-by-n_elem)
		"""
		if self.temp_variation == 'Quasi-linear':
			Factor = np.polyval(self.polyfit, t)
		elif self.temp_variation == 'bi-linear':
			Factor = np.where(t<=0,
					 			np.polyval(self.polyfit_1, t), 
								np.polyval(self.polyfit_2, t))
			
		self.E_modulus = np.ones((self.n_elements,)) * self.E_modulus_ * Factor
		self.I_moment = np.ones((self.n_elements,)) * self.I_moment_
		self.rho_density = np.ones((self.n_elements,)) * self.rho_density_
		self.A_area = np.ones((self.n_elements,)) * self.A_area_

		if 	damage_ele is not None:
			for damage_ele_, damage_severity_ in zip(damage_ele, damage_severity):
				self.E_modulus[damage_ele_] = self.E_modulus[damage_ele_]*damage_severity_
			
	def assemble_global_matrices(self):
		"""
		Assembles element matrices to create the global stiffness and mass matrices.
		
		Parameters:
		(Inputs are the physical properties and discretization info of the beam)

		Returns:
		K_global (np.ndarray): Global stiffness matrix
		M_global (np.ndarray): Global mass matrix
		"""

		self.n_nodes = self.n_elements + 1
		self.total_dofs = self.n_nodes * 2
		self.L_element = self.L_total_beam / self.n_elements

		# Initialize global matrices
		self.K_global = np.zeros((self.total_dofs, self.total_dofs))
		self.M_global = np.zeros((self.total_dofs, self.total_dofs))

		print("Assembling global stiffness and mass matrices...")
		# Iterate over each element to assemble global matrices
		for i in range(self.n_elements):
			k_e, m_e = self.get_beam_element_matrices(self.E_modulus[i], self.I_moment[i], self.rho_density[i], self.A_area[i], self.L_element)

			# Map element DOFs to global DOF indices
			dof_map = [2*i, 2*i + 1, 2*i + 2, 2*i + 3]

			# Add element matrix values to the correct positions in the global matrices
			for row_local in range(4):
				for col_local in range(4):
					row_global = dof_map[row_local]
					col_global = dof_map[col_local]
					self.K_global[row_global, col_global] += k_e[row_local, col_local]
					self.M_global[row_global, col_global] += m_e[row_local, col_local]

	def apply_boundary_conditions(self, boundary_condition='simply-supported'):
		"""
		Apply B/C to the global matrices for the eigenvalue problem (Modal analysis)

		Parameters:
		K_global (np.ndarray): Global stiffness matrix
		M_global (np.ndarray): Global mass matrix
		n_elements (int): Number of elements
		boundary_condition (str): The boundary condition

		Returns:
		M_reduced (np.ndarray): Reduced global mass matrix based on B/C
		K_reduced (np.ndarray): Reduced global stiffness matrix based on B/C
		total_dofs (int): Total Degrees of Freedom (all possible movements and rotations of the entire beam structure before considering any supports or constraints) => the "size" of the problem.
		free_dofs (int): Free Degrees of Freedom (actual unknown movements and rotations that we need to solve for after applying the supports (boundary conditions) => the parts of the beam that are "free" to vibrate
		"""
		
		# Apply boundary conditions
		if boundary_condition == 'cantilever':
			self.fixed_dofs = [0, 1] # Left end (node 0) clamped
		elif boundary_condition == 'simply-supported':
			self.fixed_dofs = [0, self.total_dofs - 2] # Displacement fixed at both ends
		else:
			raise ValueError("Unsupported boundary condition. Use 'cantilever' or 'simply-supported'.")

		self.all_dofs = np.arange(self.total_dofs)
		self.free_dofs = np.setdiff1d(self.all_dofs, self.fixed_dofs)

		# Reduce matrices
		self.K_reduced = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
		self.M_reduced = self.M_global[np.ix_(self.free_dofs, self.free_dofs)]

	def solve_eigenvalue_problem(self):
		"""
		Solves the eigenvalue problem using the global matrices and boundary conditions.

		Parameters:
		M_reduced (np.ndarray): Reduced global mass matrix based on B/C
		K_reduced (np.ndarray): Reduced global stiffness matrix based on B/C
		total_dofs (int): Total Degrees of Freedom (all possible movements and rotations of the entire beam structure before considering any supports or constraints) => the "size" of the problem.
		free_dofs (int): Free Degrees of Freedom (actual unknown movements and rotations that we need to solve for after applying the supports (boundary conditions) => the parts of the beam that are "free" to vibrate (the degrees of freedom that are NOT fixed)

		Returns:
		frequencies_hz (np.ndarray): Natural frequencies (Hz)
		mode_shapes (np.ndarray): Normalized mode shapes
		"""

		print("Solving eigenvalue problem...")
		# Solve the generalized eigenvalue problem: Kφ = λMφ (where λ = ω²)
		self.eigenvalues, self.eigenvectors_reduced = scipy.linalg.eigh(self.K_reduced, self.M_reduced)
		print("Eigenvalue problem solved.")
		
		# Calculate natural frequencies (rad/s -> Hz)
		natural_circular_frequencies_rad_s = np.sqrt(self.eigenvalues)
		self.frequencies_hz = natural_circular_frequencies_rad_s / (2 * np.pi)

		# Reconstruct full mode shapes
		self.mode_shapes = np.zeros((self.total_dofs, len(self.free_dofs)))
		self.mode_shapes[self.free_dofs, :] = self.eigenvectors_reduced
		
		# Normalize mode shapes (so that max displacement is 1)
		for i in range(self.mode_shapes.shape[1]):
			max_val = np.max(np.abs(self.mode_shapes[:, i]))
			if max_val > 1e-9:
				self.mode_shapes[:, i] /= max_val
				
		return self.frequencies_hz, self.mode_shapes
	
	def run(self, boundary_condition='simply-supported'):
		self.assemble_global_matrices()
		self.apply_boundary_conditions(boundary_condition)
		self.frequencies_hz, self.mode_shapes = self.solve_eigenvalue_problem()

		return self.frequencies_hz

	def plot_mode_shapes(self, n_modes_to_plot=4):
		"""
		Function to visualize mode shapes
		(This function remains unchanged)
		"""
		x_coords = np.linspace(0, self.L_total, self.n_nodes)

		n_modes_available = self.mode_shapes.shape[1]
		n_modes_to_plot = min(n_modes_to_plot, n_modes_available)

		plt.figure(figsize=(15, n_modes_to_plot * 3))

		for i in range(n_modes_to_plot):
			plt.subplot(n_modes_to_plot, 1, i + 1)
			
			# Extract only vertical displacements from the mode shape (even indices)
			displacements = self.mode_shapes[0::2, i]
			
			plt.plot(x_coords, displacements, 'b-', marker='o')
			plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
			plt.title(f'Mode Shape {i+1} (Frequency: {self.frequencies_hz[i]:.2f} Hz)')
			plt.xlabel('Position along beam (m)')
			plt.ylabel('Normalized Displacement')
			plt.grid(True)
			
		plt.tight_layout()
		plt.show()
	
	@classmethod
	def get_beam_element_matrices(cls, E, I, rho, A, L_e):
		"""
		Calculates the stiffness and mass matrices for a single Bernoulli-Euler beam element.

		Parameters:
		E (float): Young's Modulus
		I (float): Second moment of area
		rho (float): Material density
		A (float): Cross-sectional area
		L_e (float): Length of the element

		Returns:
		k_e (np.ndarray): Element Stiffness Matrix (4x4)
		m_e (np.ndarray): Element Consistent Mass Matrix (4x4)
		"""
		
		EI = E * I
		mu = rho * A  # mass per unit length

		# Element Stiffness Matrix
		k_e = (EI / L_e**3) * np.array([
			[12, 6 * L_e, -12, 6 * L_e],
			[6 * L_e, 4 * L_e**2, -6 * L_e, 2 * L_e**2],
			[-12, -6 * L_e, 12, -6 * L_e],
			[6 * L_e, 2 * L_e**2, -6 * L_e, 4 * L_e**2]
		])

		# Element Consistent Mass Matrix
		m_e = (mu * L_e / 420) * np.array([
			[156, 22 * L_e, 54, -13 * L_e],
			[22 * L_e, 4 * L_e**2, 13 * L_e, -3 * L_e**2],
			[54, 13 * L_e, 156, -22 * L_e],
			[-13 * L_e, -3 * L_e**2, -22 * L_e, 4 * L_e**2]
		])
		
		return k_e, m_e

if __name__ == "__main__":
	# --- Input Parameters ---
	# Example for a steel beam
	E_modulus = 2.1e11   # Young's Modulus (Pa)
	I_moment = 8.33e-6   # Second moment of area (m^4) - e.g., for a 0.1m x 0.2m rectangular cross-section
	rho_density = 7850   # Density (kg/m^3)
	A_area = 0.02        # Cross-sectional area (m^2) - e.g., 0.1m * 0.2m

	L_total_beam = 10.0  # Total length of the beam (m)
	n_elements = 20    # Number of elements to use
	n_mode_get = 6 # first six modes

	# Define class for generating dataset for numerical simulation
	temp_variation = 'bi-linear'
	Data_generator = Generate_dataset(n_elements, temp_variation, E_modulus, I_moment, rho_density, A_area, L_total_beam)

	
	# Run simulator based on the record of temperature
	fn_temp_data = 'data/2024-6~2025-6_기후데이터.csv'
	df_temp = pd.read_csv(fn_temp_data, encoding='cp949')
	t = df_temp.iloc[:, -1].values
	print(t.shape[0])

	# Define damage information
	damage_start_ind = t.shape[0] - 300
	damage_ele = [9,10]
	damage_severity = [0.9,0.9]

	freq = []
	freq_orign = []
	for time_ind in range(t.shape[0]):
		if time_ind <= damage_start_ind:
			Data_generator.set_parameter(t[time_ind], damage_ele = None, damage_severity = None)
		else:
			Data_generator.set_parameter(t[time_ind], damage_ele, damage_severity)
		frequencies_hz = Data_generator.run()
		freq.append(list(frequencies_hz[:n_mode_get]))

	for time_ind in range(t.shape[0]):
			Data_generator.set_parameter(t[time_ind], damage_ele = None, damage_severity = None)
			frequencies_hz = Data_generator.run()
			freq_orign.append(list(frequencies_hz[:n_mode_get]))
