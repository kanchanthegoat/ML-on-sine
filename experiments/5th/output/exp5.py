'''
    Highest possible features.

    Degree: 3
    Smallest feature maximum: 3.141592653589793
    Largest feature maximum: 31.006276680299816

    Degree: 11
    Smallest feature maximum: 3.141592653589793
    Largest feature maximum: 294204.01797389047

    Degree: 13
    Smallest feature maximum: 3.141592653589793
    Largest feature maximum: 2903677.270613282

    Degree: 27
    Smallest feature maximum: 3.141592653589793
    Largest feature maximum: 26487841119103.605
'''


###5B Different ranks
'''
Degree: 11
Training MSE: 4.225014e-15
Model rank: 11
Number of features: 11
Smallest singular value: 7.502451e-01

Degree: 13
Training MSE: 3.291839e-02
Model rank: 9
Number of features: 13
Smallest singular value: 4.186479e-01
'''

###FOR 5C where we actually got our result
'''
    Training MSE: 4.225014447409966e-15
Rank: 11

Degree: 11
Singular values:
[6.86372430e+05 2.05823788e+05 7.23341079e+03 2.36999714e+03
 2.18945242e+02 8.03839608e+01 1.70429750e+01 6.82855612e+00
 3.65130277e+00 1.01864305e+00 7.50245110e-01]
Estimated rank threshold: 1.524053e-08
Reported rank: 11
Training MSE: 6.089137616107032e-19
Rank: 13

Degree: 13
Singular values:
[6.36784625e+06 1.92561565e+06 5.52531366e+04 1.80368590e+04
 1.28560392e+03 4.62131019e+02 6.64187890e+01 2.68591504e+01
 8.51439341e+00 3.42963852e+00 2.28109298e+00 5.64002209e-01
 4.18647889e-01]
Estimated rank threshold: 1.413946e-07
Reported rank: 13
NumPy matrix rank: 13
NumPy singular values: [6.36784625e+06 2.08732548e+06 5.52531366e+04 1.98702531e+04
 1.28560392e+03 5.22902932e+02 6.64187890e+01 3.23277750e+01
 8.51439341e+00 4.65352837e+00 2.28109298e+00 7.50680882e-01
 4.18647889e-01]
Scikit-learn rank: 13
Scikit-learn singular values: [6.36784625e+06 1.92561565e+06 5.52531366e+04 1.80368590e+04
 1.28560392e+03 4.62131019e+02 6.64187890e+01 2.68591504e+01
 8.51439341e+00 3.42963852e+00 2.28109298e+00 5.64002209e-01
 4.18647889e-01]
'''

#Here we found  the reason behind decrease in rank . the rank improved now
