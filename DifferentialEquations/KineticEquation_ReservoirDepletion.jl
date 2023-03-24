using DifferentialEquations, Plots, Plots.PlotMeasures

# define function to solve kinetic equations
function solveKinEq(params, P, psi0, tspan)
    gamma, gR, chi, gammaR, g, R = params

    # define the kinetic equations
    function kinEq!(y1, p, t)
        psi = y1[1]
        nR = y1[2]

        dpsidt = 1/2*(R*nR - gamma)*psi - 1im*g*abs2(psi)*psi - 1im*gR*nR*psi
        dnRdt = -(gammaR + R*abs2(psi))*nR
        return [dpsidt, dnRdt]
    end

    y1 = [0.0+0.001im, P]
    prob = ODEProblem(kinEq!, y1, tspan, params)
    sol = solve(prob, tstops=t)
    psi = sol[1,:]
    nR = sol[2,:]
    return abs2.(psi), nR
end

# define simulation parameters
gamma = 0.06
gR = 1.0
chi = 0.5
gammaR = 0.0083
g = 0.5
R = 0.12
psi0 = 0.001 + 1im*0.001

P1 = 3
P2 = 6
P3 = 9

tspan = (0.0, 150.0)
t = range(tspan[1], tspan[2], length=1500)

params = [gamma, gR, chi, gammaR, g, R]
sol1 = solveKinEq(params, P1, psi0, tspan)
sol2 = solveKinEq(params, P2, psi0, tspan)
sol3 = solveKinEq(params, P3, psi0, tspan)

# plot the results
plot1 = plot(t, sol1[1], label="Small power", legend=:topright)
plot1 = plot!(t, sol2[1], label="Medium power")
plot1 = plot!(t, sol3[1], label="Large power")
plot1 = plot!(xlabel="Time (ps)", ylabel="Occupation", title="Condensate", grid=false)

plot2 = plot(t, real.(sol1[2]), label="Small power", legend=:topright)
plot2 = plot!(t, real.(sol2[2]), label="Medium power")
plot2 = plot!(t, real.(sol3[2]), label="Large power")
plot2 = plot!(xlabel="Time (ps)", ylabel="Occupation", title="Exciton reservoir", grid=false)

plot(plot1, plot2, layout=(1,2), size=(850,400), margin=5mm)
