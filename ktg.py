import math, cmath
import plotly.graph_objects as go

fig = go.Figure(go.Scattersmith(imag=[0.5, 1, 2, 3], real=[0.5, 1, 2, 3]))
fig.show()
S11 = cmath.rect(0.869, math.radians(-159))
S12 = cmath.rect(0.031, math.radians(-9))
S21 = cmath.rect(4.25, math.radians(61))
S22 = cmath.rect(0.507, math.radians(-117))
delta = S11*S22-S12*S21
cl = (S22-delta*(S11.conjugate())).conjugate()/(abs(S22)**2-abs(delta)**2)
rl = abs(S12*S21/(abs(S22)**2-abs(delta)**2))
print(f"delta: {abs(delta)}")
print(f"rl: {rl}")
print(f"cl: {cmath.polar(cl)[0]},{cmath.polar(cl)[1]*180/math.pi} deg")
print(abs(cl)-rl)

mu = (1-abs(S11)**2)/(abs(S22-delta*(S11.conjugate())) + abs(S12*S21))
print(f"mu: {mu}")
# gammaL= 1/3
# gammaS = -1/3
# gammaIN = S11+ S12*S21*gammaL/(1-S22*gammaL)
# print(f"gammaIN: {cmath.polar(gammaIN)}")
# gammaOUT = S22 + S12*S21*gammaS/(1-S11*gammaS)

# print(f"gammaOUT: {gammaOUT}")

# G = abs(S21)**2*(1-abs(gammaL)**2)/((1-abs(gammaIN)**2)*abs(1-S22*gammaL)**2)
# print(f"G: {G}")

# GA = abs(S21)**2*(1-abs(gammaS)**2)/(abs(1-S11*gammaS)**2*(1-abs(gammaOUT)**2))
# print(f"GA: {GA}")