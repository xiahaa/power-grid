# Data Generation Bugs - Root Cause Analysis

## Executive Summary
The data generation has **3 critical bugs** causing 99.7% of generated samples to have flat voltage profiles. These bugs cause loads to collapse to near-zero over time.

## Bug #1: Load Profile Applied to Already-Modified Values ⚠️ CRITICAL

### Location
`src/data/data_generator.py` lines 229-233

### Current (Broken) Code
```python
for idx, load in self.net.load.iterrows():
    base_p = self.net.load.at[idx, 'p_mw']  # ← Gets CURRENT value
    base_q = self.net.load.at[idx, 'q_mvar']
    self.net.load.at[idx, 'p_mw'] = self._generate_load_profile(base_p, hour)
    self.net.load.at[idx, 'q_mvar'] = self._generate_load_profile(base_q, hour)
```

### Problem
- At t=0: Reads original load (e.g., 0.1 MW)
- Applies profile → 0.6 * 0.1 = 0.06 MW (stored back)
- At t=1: Reads 0.06 MW (not original 0.1!)
- Applies profile again → 0.6 * 0.06 = 0.036 MW
- **Result:** Exponential decay! After 200 steps: ~0.0001 MW

### Evidence
```
t=  0: Load = 1.46 MW, V_mag std = 0.011226
t= 10: Load = 0.09 MW, V_mag std = 0.000673
t=200: Load = 0.02 MW, V_mag std = 0.000121  ← Nearly flat!
```

### Impact
- Loads collapse to near-zero
- Voltage variations disappear
- Network becomes trivial (all voltages → 1.0 p.u.)

---

## Bug #2: PV Capacity Regenerated Every Timestep ⚠️ CRITICAL

### Location
`src/data/data_generator.py` lines 236-240

### Current (Broken) Code
```python
for idx, sgen in self.net.sgen.iterrows():
    bus = self.net.sgen.at[idx, 'bus']
    bus_load = self.net.load[self.net.load.bus == bus]['p_mw'].sum()
    base_pv = bus_load * np.random.uniform(0.5, 2.0)  # ← NEW random every time!
    self.net.sgen.at[idx, 'p_mw'] = self._generate_pv_profile(base_pv, hour)
```

### Problem
- PV capacity should be **fixed per scenario** (installed capacity)
- But it's recalculated with **new random value every timestep**
- Uses collapsing `bus_load` as reference (Bug #1)
- PV capacity also collapses: 0.5-2.0x of (near-zero load) = near-zero PV

### Impact
- Unrealistic PV behavior (capacity changes every 5 minutes!)
- PV output effectively becomes zero due to collapsing base
- No meaningful generation-load interaction

---

## Bug #3: No Storage of Original Base Loads ⚠️

### Location
`src/data/data_generator.py` lines 193-202

### Current (Broken) Code
```python
def generate_scenario(self, scenario_idx: int):
    # Reset network
    self.net = self._load_network()  # ← Loads default values
    self._add_pv_systems()
    self._apply_parameter_drift(scenario_idx)

    # ... time series loop modifies self.net.load in-place
```

### Problem
- Original base loads never stored
- Time series loop modifies `self.net` in-place
- No way to recover original values for each timestep

### Impact
- Enables Bug #1 (can't reference original loads)
- No baseline for generating time-varying profiles

---

## The Cascading Failure

```
Step 1: t=0
  Original load: 0.100 MW
  Apply profile (factor 0.6): 0.060 MW
  Store back to net.load

Step 2: t=1
  Read "base" load: 0.060 MW ← Should be 0.100!
  Apply profile (factor 0.7): 0.042 MW
  Store back to net.load

Step 3: t=2
  Read "base" load: 0.042 MW ← Compounding error!
  Apply profile (factor 0.8): 0.034 MW

... continues exponentially decaying ...

Step 200: t=200
  Load ≈ 0.0001 MW
  Voltage variation ≈ 0 (all buses at 1.0 p.u.)
  Data becomes useless
```

---

## Fix Strategy

### 1. Store Original Base Loads
```python
def generate_scenario(self, scenario_idx: int):
    self.net = self._load_network()

    # Store ORIGINAL base loads before any modification
    base_loads = {
        'p_mw': self.net.load['p_mw'].values.copy(),
        'q_mvar': self.net.load['q_mvar'].values.copy()
    }

    self._add_pv_systems()

    # Store PV capacities (fixed per scenario)
    pv_capacities = self.net.sgen['p_mw'].values.copy()  # or set separately
```

### 2. Use Original Loads in Time Series Loop
```python
for t in range(self.time_steps):
    hour = (t * 5 / 60) % 24

    # Update loads from ORIGINAL base values
    for idx in range(len(self.net.load)):
        base_p = base_loads['p_mw'][idx]  # ← Use stored original
        base_q = base_loads['q_mvar'][idx]

        self.net.load.at[idx, 'p_mw'] = self._generate_load_profile(base_p, hour)
        self.net.load.at[idx, 'q_mvar'] = self._generate_load_profile(base_q, hour)
```

### 3. Fix PV Capacity (Per-Scenario Constant)
```python
def _add_pv_systems(self):
    """Add PV with FIXED capacities"""
    for bus in pv_buses:
        bus_load = self.net.load[self.net.load.bus == bus]['p_mw'].sum()
        pv_capacity = bus_load * np.random.uniform(0.5, 2.0)  # Fixed for scenario

        pp.create_sgen(
            self.net,
            bus=int(bus),
            p_mw=pv_capacity,  # Store capacity
            q_mvar=0,
            name=f"PV_{bus}"
        )

    # Store capacities
    self.pv_capacities = self.net.sgen['p_mw'].values.copy()

# Then in time series loop:
for idx, sgen in self.net.sgen.iterrows():
    base_pv = self.pv_capacities[idx]  # Use stored capacity
    self.net.sgen.at[idx, 'p_mw'] = self._generate_pv_profile(base_pv, hour)
```

---

## Testing the Fix

After implementing fixes, verify:

1. **Load stability**: Total load should oscillate around original value (±40%)
   - NOT decay to near-zero

2. **Voltage variation**: V_mag std should be 0.001-0.03 throughout
   - NOT collapse to < 1e-6

3. **Data quality**: Check samples at different timesteps:
   ```python
   t=0:   std > 0.005  ✓
   t=50:  std > 0.005  ✓
   t=200: std > 0.005  ✓  ← Should NOT be flat!
   ```

4. **PV behavior**: PV capacity should stay constant per scenario
   - Only output varies with solar profile

---

## Priority
**URGENT - P0**: This breaks the entire dataset. Must fix before any training can proceed.

## Files to Modify
1. `src/data/data_generator.py` - Implement all 3 fixes
2. `scripts/generate_data.py` - Regenerate dataset after fix
3. Verify with `scripts/diagnose_data_generation.py`
