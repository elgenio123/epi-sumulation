import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from model import sir_model, seir_model
from scipy.integrate import odeint

# Page configuration
st.set_page_config(
    page_title="Epidemic Modeling Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .parameter-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

class EpidemicSimulator:
    """
    Epidemic simulation class with placeholder methods for SciPy integration
    """
    
    def __init__(self):
        self.models = ['SIR', 'SEIR']
        self.colors = {
            'S': '#10b981',  # Green
            'E': '#f59e0b',      # Orange
            'I': '#ef4444',     # Red
            'R': '#3b82f6'     # Blue
        }
    
    def run_simulation1(self, parameters: Dict) -> Dict:
        """
        Placeholder for SciPy-based epidemic model simulation
        In production, this would integrate with scipy.integrate.odeint
        """
        
        # Extract parameters
        beta = parameters['beta']
        gamma = parameters['gamma']
        sigma = parameters.get('sigma', 0.0)
        initial_s = parameters['initial_s']
        initial_i = parameters['initial_i']
        initial_r = parameters['initial_r']
        initial_e = parameters.get('initial_e', 0)
        model_type = parameters['model_type']
        time_range = parameters['time_range']
        
        # Generate time points
        t = np.linspace(0, time_range, time_range + 1)
        
        # Mock simulation results (replace with actual SciPy integration)
        results = self._generate_mock_results(t, parameters)
        
        return {
            'time': t,
            'results': results,
            'parameters': parameters,
            'statistics': self._calculate_statistics(results)
        }
    def run_simulation(self, parameters: Dict) -> Dict:
        if parameters["model_type"] == "SIR":
            results = self._run_sir(parameters)
        elif parameters["model_type"] == "SEIR":
            results = self._run_seir(parameters)
        else:
            raise ValueError("Unknown model type")

        # Add statistics
        stats = self._calculate_statistics(results, parameters)
        results["statistics"] = stats
        return {
            "results": {k: v for k, v in results.items() if k in ["time","S", "E", "I", "R"]},
            "statistics": results["statistics"],
            "parameters": parameters,
        }
    
    
    def _run_sir(self, params: Dict) -> Dict:
        # Initial conditions
        S0 = params["initial_s"]
        I0 = params["initial_i"]
        R0 = params["initial_r"]
        N = S0 + I0 + R0
        y0 = [S0, I0, R0]

        # Time grid
        t = np.linspace(0, params["time_range"], params["time_range"]+ 1)

        # Solve ODE
        sol = odeint(sir_model, y0, t, args=(params["beta"], params["gamma"]))
        S, I, R = sol.T

        return {"time": t, "S": S, "I": I, "R": R}
    def _run_seir(self, params: Dict) -> Dict:
        # Initial conditions
        S0 = params["initial_s"]
        E0 = params["initial_e"]
        I0 = params["initial_i"]
        R0 = params["initial_r"]
        N = S0 + E0 + I0 + R0
        y0 = [S0, E0, I0, R0]

        # Time grid
        t = np.linspace(0, params["time_range"], params["time_range"])

        # Solve ODE
        sol = odeint(seir_model, y0, t, args=(params["beta"], params["sigma"], params["gamma"]))
        S, E, I, R = sol.T

        return {"time": t, "S": S, "E": E, "I": I, "R": R}


    def _calculate_statistics1(self, results: Dict) -> Dict:
        """
        Calculate key epidemic statistics
        """
        infected = results['I']
        
        return {
            'peak_infected': np.max(infected),
            'peak_day': np.argmax(infected),
            'total_cases': np.sum(np.diff(results['R'], prepend=0)),
            'attack_rate': (np.max(results['R']) / 
                          (results['S'][0] + results['I'][0] + results['R'][0])) * 100,
            'basic_reproduction_number': 2.5  # Placeholder calculation
        }
    def _calculate_statistics(self, results: Dict, parameters: Dict) -> Dict:
        I = results["I"]
        peak_infected = int(np.max(I))
        peak_day = int(results["time"][np.argmax(I)])
        total_cases = int(parameters["initial_s"] + parameters["initial_i"] + parameters["initial_r"] - results["S"][-1])
        attack_rate = (total_cases / (parameters["initial_s"] + parameters["initial_i"] + parameters["initial_r"])) * 100
        
        stats = {
            "peak_infected": peak_infected,
            "peak_day": peak_day,
            "total_cases": total_cases,
            "attack_rate": f"{attack_rate:.2f}%",
            "R0": parameters["beta"] / parameters["gamma"]
        }
        return stats

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'last_run_time' not in st.session_state:
        st.session_state.last_run_time = None

def render_header():
    """Render the main header section"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🦠 Epidemic Modeling Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced SIR/SEIR Simulation Platform for Infectious Disease Modeling</p>', 
                   unsafe_allow_html=True)
    
    with col2:
        if st.session_state.last_run_time:
            st.metric("Last Simulation", 
                     st.session_state.last_run_time.strftime("%H:%M:%S"),
                     delta="Completed")

def render_parameter_controls(simulator: EpidemicSimulator) -> Dict:
    """Render sidebar parameter controls"""
    
    st.sidebar.markdown("## 🎛️ Simulation Parameters")
    
    # Model Selection
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### Model Configuration")
    model_type = st.sidebar.selectbox(
        "Epidemic Model",
        simulator.models,
        help="Choose between different compartmental models"
    )
    
    time_range = st.sidebar.slider(
        "Simulation Time (days)",
        min_value=30,
        max_value=365,
        value=150,
        step=5,
        help="Duration of the simulation in days"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Disease Parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### Disease Parameters")
    
    beta = st.sidebar.slider(
        "Transmission Rate (β)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        help="Rate of infection transmission per contact"
    )
    
    gamma = st.sidebar.slider(
        "Recovery Rate (γ)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Rate at which infected individuals recover"
    )
    
    sigma = None
    if model_type in ['SEIR', 'SEIS']:
        sigma = st.sidebar.slider(
            "Incubation Rate (σ)",
            min_value=0.01,
            max_value=0.5,
            value=0.2,
            step=0.01,
            help="Rate at which exposed individuals become infectious"
        )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Population Parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### Initial Population")
    
    total_population = st.sidebar.number_input(
        "Total Population",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=1000,
        help="Total population size"
    )
    
    initial_i = st.sidebar.number_input(
        "Initial Infected",
        min_value=1,
        max_value=int(total_population * 0.1),
        value=100,
        step=1,
        help="Number of initially infected individuals"
    )
    
    initial_r = st.sidebar.number_input(
        "Initial Recovered",
        min_value=0,
        max_value=int(total_population * 0.5),
        value=0,
        step=1,
        help="Number of initially recovered individuals"
    )
    
    initial_e = 0
    if model_type in ['SEIR', 'SEIS']:
        initial_e = st.sidebar.number_input(
            "Initial Exposed",
            min_value=0,
            max_value=int(total_population * 0.1),
            value=0,
            step=1,
            help="Number of initially exposed individuals"
        )
    
    initial_s = total_population - initial_i - initial_r - initial_e
    st.sidebar.metric("Calculated Susceptible", f"{initial_s:,}")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Derived Parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### Derived Parameters")
    
    if gamma > 0:
        infectious_period = 1 / gamma
        st.sidebar.metric("Infectious Period", f"{infectious_period:.1f} days")
    
    if sigma and sigma > 0:
        incubation_period = 1 / sigma
        st.sidebar.metric("Incubation Period", f"{incubation_period:.1f} days")
    
    basic_r0 = beta / gamma if gamma > 0 else 0
    st.sidebar.metric("Basic R₀", f"{basic_r0:.2f}")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'model_type': model_type,
        'beta': beta,
        'gamma': gamma,
        'sigma': sigma,
        'initial_s': initial_s,
        'initial_i': initial_i,
        'initial_r': initial_r,
        'initial_e': initial_e,
        'time_range': time_range,
        'total_population': total_population
    }

def render_simulation_controls():
    """Render simulation control buttons"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_simulation = st.button(
            "🚀 Run Simulation",
            help="Execute the epidemic simulation with current parameters",
            use_container_width=True
        )
    
    with col2:
        reset_params = st.button(
            "🔄 Reset Parameters", 
            help="Reset all parameters to default values",
            use_container_width=True
        )
    
    with col3:
        export_results = st.button(
            "📊 Export Results",
            help="Download simulation results and graphs",
            use_container_width=True
        )
    
    return run_simulation, reset_params, export_results

def render_simulation_graph(results: Optional[Dict], simulator: EpidemicSimulator):
    """Render the main simulation graph"""
    
    st.markdown("## 📈 Epidemic Curves")
    
    if results is None:
        # Placeholder graph
        fig = go.Figure()
        fig.add_annotation(
            text="Click 'Run Simulation' to generate epidemic curves",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            title="Epidemic Simulation Results",
            xaxis_title="Time (days)",
            yaxis_title="Population",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Create simulation graph
    fig = go.Figure()
    #print(results)
    time_data = results['results']['time']
    simulation_results = results['results']
    
    # Add traces for each compartment
    for compartment, data in simulation_results.items():
        if compartment == "time":
            continue 
        fig.add_trace(go.Scatter(
            x=time_data,
            y=data,
            mode='lines',
            name=compartment.title(),
            line=dict(
                color=simulator.colors.get(compartment, '#6b7280'),
                width=3
            ),
            hovertemplate=f"<b>{compartment.title()}</b><br>" +
                         "Day: %{x}<br>" +
                         "Population: %{y:,.0f}<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title={
            'text': f"Epidemic Simulation - {results['parameters']['model_type']} Model",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Time (days)",
        yaxis_title="Population",
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.8)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_statistics_panel(results: Optional[Dict]):
    """Render simulation statistics and key metrics"""
    
    st.markdown("## 📊 Simulation Statistics")
    
    if results is None:
        st.info("Run a simulation to see detailed statistics and key metrics.")
        return
    
    stats = results['statistics']
    params = results['parameters']
    print('Debug',stats)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Peak Infected",
            f"{stats['peak_infected']:,.0f}",
            delta=f"Day {stats['peak_day']}"
        )
    
    with col2:
        st.metric(
            "Attack Rate",
            #f"{:.1f}%" ,
            stats['attack_rate'],
            delta=f"{stats['total_cases']:,.0f} total cases"
        )
    
    '''with col3:
        st.metric(
            "Basic R₀",
            #f"{:.2f}",
            stats['basic_reproduction_number'],
            delta="Reproduction number"
        )'''
    
    with col4:
        infectious_period = 1 / params['gamma'] if params['gamma'] > 0 else 0
        st.metric(
            "Infectious Period",
            f"{infectious_period:.1f} days",
            delta="Average duration"
        )
    
    # Additional Statistics
    st.markdown("### Detailed Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Configuration:**")
        st.write(f"• Model Type: {params['model_type']}")
        st.write(f"• Transmission Rate (β): {params['beta']:.3f}")
        st.write(f"• Recovery Rate (γ): {params['gamma']:.3f}")
        if params['sigma']:
            st.write(f"• Incubation Rate (σ): {params['sigma']:.3f}")
    
    with col2:
        st.markdown("**Population Dynamics:**")
        st.write(f"• Total Population: {params['total_population']:,}")
        st.write(f"• Initial Susceptible: {params['initial_s']:,}")
        st.write(f"• Initial Infected: {params['initial_i']:,}")
        st.write(f"• Initial Recovered: {params['initial_r']:,}")
        if params['model_type'] in ['SEIR', 'SEIS']:
            st.write(f"• Initial Exposed: {params['initial_e']:,}")

def export_simulation_results(results: Optional[Dict]):
    """Handle export functionality"""
    if results is None:
        st.warning("No simulation results to export. Please run a simulation first.")
        return
    
    # Create DataFrame for export
    df_data = {
        'Time': results['results']['time'],
        'Susceptible': results['results']['S'],
        'Infected': results['results']['I'],
        'Recovered': results['results']['R']
    }
    
    if 'E' in results['results']:
        df_data['Exposed'] = results['results']['E']
    
    df = pd.DataFrame(df_data)
    
    # Convert to CSV
    csv_data = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_data,
        file_name=f"epidemic_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize simulator
    simulator = EpidemicSimulator()
    
    # Render header
    render_header()
    
    # Render parameter controls in sidebar
    parameters = render_parameter_controls(simulator)
    
    # Render simulation controls
    run_sim, reset_params, export_results = render_simulation_controls()
    
    # Handle button actions
    if run_sim:
        with st.spinner("Running epidemic simulation..."):
            # Simulate processing time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate computation
                progress_bar.progress(i + 1)
            
            # Run simulation
            results = simulator.run_simulation(parameters)
            st.session_state.simulation_results = results
            st.session_state.last_run_time = datetime.now()
            
            st.success("Simulation completed successfully!")
            time.sleep(1)
            st.rerun()
    
    if reset_params:
        st.session_state.simulation_results = None
        st.session_state.last_run_time = None
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Render simulation graph
        render_simulation_graph(st.session_state.simulation_results, simulator)
    
    with col2:
        # Render statistics panel
        render_statistics_panel(st.session_state.simulation_results)
    
    # Export functionality
    if export_results:
        st.markdown("## 📤 Export Results")
        export_simulation_results(st.session_state.simulation_results)

if __name__ == "__main__":
    main()
