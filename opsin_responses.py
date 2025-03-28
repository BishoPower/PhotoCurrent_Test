import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cleo
    import brian2.only as b2
    from brian2 import np, ms, namp

    b2.prefs.codegen.target = "numpy"
    return b2, cleo, mo, ms, namp, np


@app.cell
def _(b2, cleo):
    ng = b2.NeuronGroup(
        1,
        """v = -70*mV : volt
        Iopto : amp
        x = 0*um : meter
        y = 0*um : meter
        z = 0*um : meter""",
    )
    mon = b2.StateMonitor(ng, "Iopto", record=True)
    net = b2.Network(ng, mon)
    sim = cleo.CLSimulator(net)

    chr2 = cleo.opto.chr2_4s()
    sim.inject(chr2, ng)

    light = cleo.light.Light(
        coords=[0, 0, -100] * b2.um, light_model=cleo.light.OpticFiber()
    )
    sim.inject(light, ng)
    return chr2, light, mon, net, ng, sim


@app.cell
def _(ms):
    t_pre_stim = 10 * ms
    impulse_width = 5 * ms
    pulse_width = 300 * ms
    t_post_stim = 200 * ms
    return impulse_width, pulse_width, t_post_stim, t_pre_stim


@app.cell
def _(b2, impulse_width, light, mon, ms, namp, sim, t_post_stim, t_pre_stim):
    import pandas as pd

    # impulse response
    sim.run(t_pre_stim)
    u_impulse = 100
    light.update(u_impulse * b2.mwatt / b2.mm2)

    sim.run(impulse_width)
    light.update(0)
    sim.run(t_post_stim)

    impulse_data = pd.DataFrame(
        {
            "t_ms": mon.t / ms,
            "I_nA": mon.Iopto[0] / namp,
            "u_mWmm2": u_impulse
            * ((mon.t >= t_pre_stim) & (mon.t < t_pre_stim + impulse_width)),
        }
    )

    sim.reset()
    return impulse_data, pd, u_impulse


@app.cell
def _(b2, light, mon, ms, namp, pd, pulse_width, sim, t_post_stim, t_pre_stim):
    # pulse response
    sim.run(t_pre_stim)
    u_pulse = 20
    light.update(u_pulse * b2.mwatt / b2.mm2)

    sim.run(pulse_width)
    light.update(0)
    sim.run(t_post_stim)

    pulse_data = pd.DataFrame(
        {
            "t_ms": mon.t / ms,
            "I_nA": mon.Iopto[0] / namp,
            "u_mWmm2": u_pulse
            * ((mon.t >= t_pre_stim) & (mon.t < t_pre_stim + pulse_width)),
        }
    )

    sim.reset()
    return pulse_data, u_pulse


@app.cell
def _(impulse_data, light):
    import matplotlib.pyplot as plt


    def plot_data(data):
        fig, axs = plt.subplots(2, 1, layout="constrained", sharex=True)
        axs[0].plot(data["t_ms"], data["I_nA"])
        axs[0].set(ylabel="$I_{opto}$ (nA)")
        axs[1].plot(data["t_ms"], data["u_mWmm2"], c=light.color)
        axs[1].set(ylabel="$u$ (mW/mm²)", xlabel="t (ms)")
        plt.show()


    plot_data(impulse_data)
    return plot_data, plt


@app.cell
def _(pulse_data):
    pulse_data.to_csv("pulse_data.csv", index=False)
    return


@app.cell
def _(plot_data, pulse_data):
    plot_data(pulse_data)
    return


if __name__ == "__main__":
    app.run()
