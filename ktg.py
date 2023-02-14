import math, cmath

S11 = cmath.rect(0.869, math.radians(-159))
S12 = cmath.rect(0.031, math.radians(-9))
S21 = cmath.rect(4.25, math.radians(61))
S22 = cmath.rect(0.507, math.radians(-117))
delta = S11*S22-S12*S21
cl = (S22-delta*(S11.conjugate())).conjugate()/(abs(S22)**2-abs(delta)**2)
rl = abs(S12*S21/(abs(S22)**2-abs(delta)**2))
print(rl)
print(cl)
print(abs(cl)-rl)

mu = (1-abs(S11)**2)/(abs(S22-delta*(S11.conjugate())) + abs(S12*S21))
print(f"mu: {mu}")