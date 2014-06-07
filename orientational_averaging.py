#! /usr/bin/env python

from __future__ import print_function, division
import sys, os

def main():
    """\
    This program will help with the evaluation of orientational averages by
    showing which permutations of indices are unique.

    Currently this is very basic but it is meant to be simple.

    I follow from the expressions given in:
    D. L. Andrews and T. Thirunamachandran.  J. Chem. Phys. 67, 5026, 1977.
    D. L. Andrews and W. A. Ghoul.  J. Phys. A: Math. Gen. 14, 1281, 1981.
    """

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()

    import itertools
    from numpy import zeros, matrix, linalg, array

    # Create a file ('oa' for orientational average)
    fh_pol = open('oa_raman.data', 'w')
    fh_hpol = open('oa_hyperraman.data', 'w')
    fh_2hpol = open('oa_secondhyperraman.data', 'w')

    # +++++ Polarizability +++++

    # For the polarizability, we are concerned with the average:
    #   <alpha_ij^2> = sum_{ab,cd}[ <T_ia*T_jb*T_ic*T_jd> alpha_ab*alpha_cd ]
    #
    # For Raman scattering measured in a perpendicular orientation, we need
    # the averages <alpha_ii^2> and <alpha_ij^2>.  For averaging of the 4th
    # rank tensor on the right side of the equation, only two circumstances
    # give nonzero averages:
    #   1. a = b = c = d
    #   2. a = b, c = d
    # These are stored in the lists below.
    #laaaa = ['a', 'a', 'a', 'a']
    #laabb = ['a', 'a', 'b', 'b']
    laaaa = [1, 1, 1, 1]
    laabb = [1, 1, 2, 2]

    saaaa = set()
    saabb = set()

    genaaaa = itertools.permutations(laaaa,4)
    genaabb = itertools.permutations(laabb,4)

    txt = 'Polarizability Averaging Indices'
    print(len(txt)*'+', file=fh_pol)  
    print(txt, file=fh_pol)
    print(len(txt)*'+', file=fh_pol)

    # Size of the basis set and number of linearly independent terms
    r4nn, r4qn = fullpermutations(4)
    print('', file=fh_pol)
    txt = 'For a tensor of rank 4'
    print('*'*2*len(txt), file=fh_pol)
    print(txt, file=fh_pol)
    print('*'*2*len(txt), file=fh_pol)
    txt = 'Size of basis set = ' + str(r4nn)
    print(txt, file=fh_pol)
    txt = 'Number of linearly independent terms = ' + str(r4qn)
    print(txt, file=fh_pol)
    print('', file=fh_pol)

    # Terms with aa,aa
    txt = 'Indices for aa,aa terms'
    print(len(txt)*'=', file=fh_pol)
    print(txt, file=fh_pol)
    print(len(txt)*'=', file=fh_pol)
    for i in genaaaa:
        if i not in saaaa:
            print(i, file=fh_pol)
        saaaa.add(i)

    print('', file=fh_pol)
    # Terms with aa,bb (all permutations)
    txt = 'Indices for aa,bb terms'
    print(len(txt)*'=', file=fh_pol)
    print(txt, file=fh_pol)
    print(len(txt)*'=', file=fh_pol)
    for i in genaabb:
        if i not in saabb:
            print(i, file=fh_pol)
        saabb.add(i)

    print('', file=fh_pol)
    print('~'*30, file=fh_pol)
    print('Number of aa,aa terms', len(saaaa), file=fh_pol)
    print('Number of aa,bb terms', len(saabb), file=fh_pol)
    print('~'*30, file=fh_pol)
    print('', file=fh_pol)

    # Terms for Mathematica
    print('%'*30, file=fh_pol)
    print('Mathematica style output', file=fh_pol)
    print('%'*30, file=fh_pol)    
    print('', file=fh_pol)

    # Basis vectors in the experimental reference frame
    r4exp, r4mol = vectors_exp_mol(4)
    print('Experimental reference frame basis vectors', file=fh_pol)
    for item in r4exp:
        print(item, file=fh_pol)
    print('', file=fh_pol)

    # Matrix for generating orientational averages
    smat, vexp, vmol = generate_smat_and_vecs(r4nn,4,False)
    print('S matrix', file=fh_pol)
    print(smat, file=fh_pol)
    print('', file=fh_pol)
    
    # Basis vectors in the molecular reference frame
    print('Molecular reference frame basis vectors', file=fh_pol)
    for item in r4mol:
        print(item, file=fh_pol)
    print('', file=fh_pol)

    # Experimental vector containing basis vectors
    print('Experimental total vector', file=fh_pol)
    print(vexp, file=fh_pol)
    print('', file=fh_pol)

    # Molecular vector containing basis vectors
    print('Molecular total vector', file=fh_pol)
    print(vmol, file=fh_pol)
    print('', file=fh_pol)

    # Index equivalence for molecular reference frame data
    data, avg_alphaii, avg_alphaij = pol_mathematica(saaaa, saabb)  

    print('Index equivalence for molecular reference frame vectors', file=fh_pol)
    for item in data:
        print(item, file=fh_pol)
    print('', file=fh_pol)

    print('Polarizability Average Terms', file=fh_pol)
    print('<alpha_ii^2> term', file=fh_pol)
    for item in avg_alphaii:
        print(item, file=fh_pol)
    print('', file=fh_pol)
    print('<alpha_ij^2> term', file=fh_pol)
    for item in avg_alphaij:
        print(item, file=fh_pol)

    # +++++ First Hyperpolarizability +++++

    # For the first hyperpolarizability, we are concerned with the average:
    #   <beta_ijk^2> 
    #           = sum_{abc,def}[ <T_ia*T_jb*T_kc*T_id*T_je*T_kf> beta_abc*beta_def ]
    #
    # For hyper-Raman scattering measured in a perpendicular orientation, we need
    # the averages <beta_iii^2> and <beta_ijj^2>.  For averaging of the 6th
    # rank tensor on the right side of the equation, three circumstances
    # give nonzero averages:
    #   1. a = b = c = d = e = f
    #   2. a = b = c = d, e = f
    #   3. a = b, c = d, e = f
    # These are stored in the lists below.
    #laaaaaa = ['a', 'a', 'a', 'a', 'a', 'a']
    #laaaabb = ['a', 'a', 'a', 'a', 'b', 'b']
    #laabbcc = ['a', 'a', 'b', 'b', 'c', 'c']
    laaaaaa = [1, 1, 1, 1, 1, 1]
    laaaabb = [1, 1, 1, 1, 2, 2]
    laabbcc = [1, 1, 2, 2, 3, 3]

    saaaaaa = set()
    saaaabb = set()
    saabbcc = set()

    genaaaaaa = itertools.permutations(laaaaaa,6)
    genaaaabb = itertools.permutations(laaaabb,6)
    genaabbcc = itertools.permutations(laabbcc,6)

    txt = 'First hyperpolarizability Averaging Indices'
    print(len(txt)*'+', file=fh_hpol)  
    print(txt, file=fh_hpol)
    print(len(txt)*'+', file=fh_hpol)

    # Size of the basis set and number of linearly independent terms
    r6nn, r6qn = fullpermutations(6)
    print('', file=fh_hpol)
    txt = 'For a tensor of rank 6'
    print('*'*2*len(txt), file=fh_hpol)
    print(txt, file=fh_hpol)
    print('*'*2*len(txt), file=fh_hpol)
    txt = 'Size of basis set = ' + str(r6nn)
    print(txt, file=fh_hpol)
    txt = 'Number of linearly independent terms = ' + str(r6qn)
    print(txt, file=fh_hpol)
    print('', file=fh_hpol)

    # Terms with aaa,aaa
    txt = 'Indices for aaa,aaa terms'
    print(len(txt)*'=', file=fh_hpol)
    print(txt, file=fh_hpol)
    print(len(txt)*'=', file=fh_hpol)
    for i in genaaaaaa:
        if i not in saaaaaa:
            print(i, file=fh_hpol)
        saaaaaa.add(i)

    print('', file=fh_hpol)
    # Terms with aaa,abb (all permutations)
    txt = 'Indices for aaa,abb terms'
    print(len(txt)*'=', file=fh_hpol)
    print(txt, file=fh_hpol)
    print(len(txt)*'=', file=fh_hpol)
    for i in genaaaabb:
        if i not in saaaabb:
            print(i, file=fh_hpol)
        saaaabb.add(i)

    print('', file=fh_hpol)
    # Terms with aab,bcc (all permutations)
    # Here, we need to be careful that we don't overcount terms.  It
    # is very easy to come up with an overcomplete basis.
    txt = 'Indices for aab,bcc terms'
    print(len(txt)*'=', file=fh_hpol)
    print(txt, file=fh_hpol)
    print(len(txt)*'=', file=fh_hpol)

    # This will generate all combinations of the aab,bcc terms.  However,
    # it requires more prior knowledge than I'd like. 
    #count1 = 0
    #count2 = 0
    #count3 = 0
    #count4 = 0
    #count5 = 0
    #for i in genaabbcc:
    #    if i not in saabbcc:
    #        if i[1] == 1:
    #            count1 +=1
    #            if count1 <= 3:
    #                print(i, file=fh_hpol)
    #                saabbcc.add(i)
    #        elif i[2] == 1:
    #            count2 +=1
    #            if count2 <= 3:
    #                print(i, file=fh_hpol)
    #                saabbcc.add(i)
    #        elif i[3] == 1:
    #            count3 +=1
    #            if count3 <= 3:
    #                print(i, file=fh_hpol)
    #                saabbcc.add(i)
    #        elif i[4] == 1:
    #            count4 +=1
    #            if count4 <= 3:
    #                print(i, file=fh_hpol)
    #                saabbcc.add(i)
    #        elif i[5] == 1:
    #            count5 +=1
    #            if count5 <= 3:
    #                print(i, file=fh_hpol)
    #                saabbcc.add(i)
    # Generate all combinations of aab,bcc terms.  We remove the redundant
    # elements next.
    for i in genaabbcc:
        if i not in saabbcc:
            saabbcc.add(i)

    # Basis functions of Kronecker delta products
    f61m  = "krond(a,b)*krond(c,d)*krond(e,f)"
    f62m  = "krond(a,b)*krond(c,e)*krond(d,f)"
    f63m  = "krond(a,b)*krond(c,f)*krond(d,e)"
    f64m  = "krond(a,c)*krond(b,d)*krond(e,f)"
    f65m  = "krond(a,c)*krond(b,e)*krond(d,f)"
    f66m  = "krond(a,c)*krond(b,f)*krond(d,e)"
    f67m  = "krond(a,d)*krond(b,c)*krond(e,f)"
    f68m  = "krond(a,d)*krond(b,e)*krond(c,f)"
    f69m  = "krond(a,d)*krond(b,f)*krond(c,e)"
    f610m = "krond(a,e)*krond(b,c)*krond(d,f)"
    f611m = "krond(a,e)*krond(b,d)*krond(c,f)"
    f612m = "krond(a,e)*krond(b,f)*krond(c,d)"
    f613m = "krond(a,f)*krond(b,c)*krond(d,e)"
    f614m = "krond(a,f)*krond(b,d)*krond(c,e)"
    f615m = "krond(a,f)*krond(b,e)*krond(c,d)"

    lmol = [ f61m,  f62m,  f63m,  f64m,  f65m, 
             f66m,  f67m,  f68m,  f69m,  f610m,
             f611m, f612m, f613m, f614m, f615m ]

    # Temporary set for checking uniqueness
    stmp = set()
    # This set stores the elements of saabbcc that are redundant when 
    # we insert values of the indices.
    sintersect = set()
    # Loop through the elements of saabbcc
    for item in saabbcc:
        # Assign values to the indices
        a = item[0]
        b = item[1]
        c = item[2]
        d = item[3]
        e = item[4]
        f = item[5]
        # Temporary list for storing vectors with values
        tmp = []
        for vec in lmol:
            # Evaluate the value of the Kronecker delta products
            v = eval_krond(vec, a, b, c, d, e, f, 0, 0)
            tmp.append(v)
        # We need immutable objects to add in a set
        tmp = tuple(tmp)
        # Add to sintersect if the item is in stmp
        if tmp in stmp:
            sintersect.add(item)
        # Add to stmp if it isn't present
        else:
            stmp.add(tmp)
    # This function removes elements of saabbcc that intersect with
    # elements of sintersect.  The result is a set containing only the 
    # unique elements.
    saabbcc.difference_update(sintersect)

    # Print elements of saabbcc.
    for i in saabbcc:
        print(i, file=fh_hpol)

    print('', file=fh_hpol)
    print('~'*30, file=fh_hpol)
    print('Number of aaa,aaa terms', len(saaaaaa), file=fh_hpol)
    print('Number of aaa,abb terms', len(saaaabb), file=fh_hpol)
    print('Number of aab,bcc terms', len(saabbcc), file=fh_hpol)
    print('~'*30, file=fh_hpol)
    print('', file=fh_hpol)

    print('%'*30, file=fh_hpol)
    print('Mathematica style output', file=fh_hpol)
    print('%'*30, file=fh_hpol)
    print('', file=fh_hpol)

    # Basis vectors in the experimental reference frame
    r6exp, r6mol = vectors_exp_mol(6)
    print('Experimental reference frame basis vectors', file=fh_hpol)
    for item in r6exp:
        print(item, file=fh_hpol)
    print('', file=fh_hpol)

    # Matrix for generating orientational averages
    smat, vexp, vmol = generate_smat_and_vecs(r6nn,6,False)
    print('S matrix', file=fh_hpol)
    print(smat, file=fh_hpol)
    print('', file=fh_hpol)

    # Basis vectors in the molecular reference frame
    print('Molecular reference frame basis vectors', file=fh_hpol)
    for item in r6mol:
        print(item, file=fh_hpol)
    print('', file=fh_hpol)

    # Experimental vector containing basis vectors
    print('Experimental total vector', file=fh_hpol)
    print(vexp, file=fh_hpol)
    print('', file=fh_hpol)

    # Molecular vector containing basis vectors
    print('Molecular total vector', file=fh_hpol)
    print(vmol, file=fh_hpol)
    print('', file=fh_hpol)

    data, avg_betaiii, avg_betaijj = hpol_mathematica(saaaaaa, saaaabb, saabbcc)

    print('Set up molecular reference frame vectors', file=fh_hpol)
    for item in data:
        print(item, file=fh_hpol)
    print('', file=fh_hpol)

    print('Hyperpolarizability Average Terms', file=fh_hpol)
    print('<beta_iii^2> term', file=fh_hpol)
    for item in avg_betaiii:
        print(item, file=fh_hpol)
    print('', file=fh_hpol)
    print('<beta_ijj^2> term', file=fh_hpol)
    for item in avg_betaijj:
        print(item, file=fh_hpol)

    # +++++ Second Hyperpolarizability +++++

    # For the second hyperpolarizability, we are concerned with the average:
    #   <gamma_ijkl^2> 
    #           = sum_{abcd,efgh}[ <T_ia*T_jb*T_kc*T_ld*T_ie*T_jf*T_kg*T_lh> gamma_abcd*gamma_efgh ]
    #
    # For second hyper-Raman scattering measured in a perpendicular orientation, we need
    # the averages <gamma_iiii^2> and <gamma_ijjj^2>.  For averaging of the 8th
    # rank tensor on the right side of the equation, four circumstances
    # give nonzero averages:
    #   1. a = b = c = d = e = f = g = h
    #   2. a = b = c = d = e = f, g = h
    #   3. a = b = c = d, e = f = g = h
    #   4. a = b = c = d, e = f, g = h
    # These are stored in the lists below.
    #laaaaaaaa = ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
    #laaaaaabb = ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b']
    #laaaabbbb = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
    #laaaabbcc = ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c']
    laaaaaaaa = [1, 1, 1, 1, 1, 1, 1, 1]
    laaaaaabb = [1, 1, 1, 1, 1, 1, 2, 2]
    laaaabbbb = [1, 1, 1, 1, 2, 2, 2, 2]
    laaaabbcc = [1, 1, 1, 1, 2, 2, 3, 3]
    # This type of average is actually equivalent to the fourth term,
    # because the indices can only be x, y, or z.  
    #laabbccdd = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']

    saaaaaaaa = set()
    saaaaaabb = set()
    saaaabbbb = set()
    saaaabbcc = set()
    #saabbccdd = set()

    genaaaaaaaa = itertools.permutations(laaaaaaaa,8)
    genaaaaaabb = itertools.permutations(laaaaaabb,8)
    genaaaabbbb = itertools.permutations(laaaabbbb,8)
    genaaaabbcc = itertools.permutations(laaaabbcc,8)
    #genaabbccdd = itertools.permutations(laabbccdd,8)

    txt = 'Second hyperpolarizability Averaging Indices'
    print(len(txt)*'+', file=fh_2hpol)
    print(txt, file=fh_2hpol)
    print(len(txt)*'+', file=fh_2hpol)

    # Size of the basis set and number of linearly independent terms
    r8nn, r8qn = fullpermutations(8)
    print('', file=fh_2hpol)
    txt = 'For a tensor of rank 8'
    print('*'*2*len(txt), file=fh_2hpol)
    print(txt, file=fh_2hpol)
    print('*'*2*len(txt), file=fh_2hpol)
    txt = 'Size of basis set = ' + str(r8nn)
    print(txt, file=fh_2hpol)
    txt = 'Number of linearly independent terms = ' + str(r8qn)
    print(txt, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Terms with aaaa,aaaa
    txt = 'Indices for aaaa,aaaa terms'
    print(len(txt)*'=', file=fh_2hpol)
    print(txt, file=fh_2hpol)
    print(len(txt)*'=', file=fh_2hpol)
    for i in genaaaaaaaa:
        if i not in saaaaaaaa:
            print(i, file=fh_2hpol)
        saaaaaaaa.add(i)

    print('', file=fh_2hpol)
    # Terms with aaaa,aabb (all permutations)
    txt = 'Indices for aaaa,aabb terms'
    print(len(txt)*'=', file=fh_2hpol)
    print(txt, file=fh_2hpol)
    print(len(txt)*'=', file=fh_2hpol)
    for i in genaaaaaabb:
        if i not in saaaaaabb:
            print(i, file=fh_2hpol)
        saaaaaabb.add(i)

    print('', file=fh_2hpol)
    # Terms with aaaa,bbbb (all permutations)
    txt = 'Indices for aaaa,bbbb terms'
    print(len(txt)*'=', file=fh_2hpol)
    print(txt, file=fh_2hpol)
    print(len(txt)*'=', file=fh_2hpol)
    for i in genaaaabbbb:
        if i not in saaaabbbb:
            print(i, file=fh_2hpol)
        saaaabbbb.add(i)

    print('', file=fh_2hpol)
    # Terms with aaaa,bbcc (all permutations)
    txt = 'Indices for aaaa,bbcc terms'
    print(len(txt)*'=', file=fh_2hpol)
    print(txt, file=fh_2hpol)
    print(len(txt)*'=', file=fh_2hpol)
    # Temporarily, we store all permutations.  There are 420 permutations,
    # but only 210 of them are unique.
    for i in genaaaabbcc:
        if i not in saaaabbcc:
            #print(i, file=fh_2hpol)
            saaaabbcc.add(i)

    # Set up the Kronecker delta products as strings. 
    f81m   = 'krond(a,b)*krond(c,d)*krond(e,f)*krond(g,h)'
    f82m   = 'krond(a,b)*krond(c,d)*krond(e,g)*krond(f,h)'
    f83m   = 'krond(a,b)*krond(c,d)*krond(e,h)*krond(f,g)'
    f84m   = 'krond(a,b)*krond(c,e)*krond(d,f)*krond(g,h)'
    f85m   = 'krond(a,b)*krond(c,e)*krond(d,g)*krond(f,h)'
    f86m   = 'krond(a,b)*krond(c,e)*krond(d,h)*krond(f,g)'
    f87m   = 'krond(a,b)*krond(c,f)*krond(d,e)*krond(g,h)'
    f88m   = 'krond(a,b)*krond(c,f)*krond(d,g)*krond(e,h)'
    f89m   = 'krond(a,b)*krond(c,f)*krond(d,h)*krond(e,g)'
    f810m  = 'krond(a,b)*krond(c,g)*krond(d,e)*krond(f,h)'
    f811m  = 'krond(a,b)*krond(c,g)*krond(d,f)*krond(e,h)'
    f812m  = 'krond(a,b)*krond(c,g)*krond(d,h)*krond(e,f)'
    f813m  = 'krond(a,b)*krond(c,h)*krond(d,e)*krond(f,g)'
    f814m  = 'krond(a,b)*krond(c,h)*krond(d,f)*krond(e,g)'
    f815m  = 'krond(a,b)*krond(c,h)*krond(d,g)*krond(e,f)'
    f816m  = 'krond(a,c)*krond(b,d)*krond(e,f)*krond(g,h)'
    f817m  = 'krond(a,c)*krond(b,d)*krond(e,g)*krond(f,h)'
    f818m  = 'krond(a,c)*krond(b,d)*krond(e,h)*krond(f,g)'
    f819m  = 'krond(a,c)*krond(b,e)*krond(d,f)*krond(g,h)'
    f820m  = 'krond(a,c)*krond(b,e)*krond(d,g)*krond(f,h)'
    f821m  = 'krond(a,c)*krond(b,e)*krond(d,h)*krond(f,g)'
    f822m  = 'krond(a,c)*krond(b,f)*krond(d,e)*krond(g,h)'
    f823m  = 'krond(a,c)*krond(b,f)*krond(d,g)*krond(e,h)'
    f824m  = 'krond(a,c)*krond(b,f)*krond(d,h)*krond(e,g)'
    f825m  = 'krond(a,c)*krond(b,g)*krond(d,e)*krond(f,h)'
    f826m  = 'krond(a,c)*krond(b,g)*krond(d,f)*krond(e,h)'
    f827m  = 'krond(a,c)*krond(b,g)*krond(d,h)*krond(e,f)'
    f828m  = 'krond(a,c)*krond(b,h)*krond(d,e)*krond(f,g)'
    f829m  = 'krond(a,c)*krond(b,h)*krond(d,f)*krond(e,g)'
    f830m  = 'krond(a,c)*krond(b,h)*krond(d,g)*krond(e,f)'
    f831m  = 'krond(a,d)*krond(b,c)*krond(e,f)*krond(g,h)'
    f832m  = 'krond(a,d)*krond(b,c)*krond(e,g)*krond(f,h)'
    f833m  = 'krond(a,d)*krond(b,c)*krond(e,h)*krond(f,g)'
    f834m  = 'krond(a,d)*krond(b,e)*krond(c,f)*krond(g,h)'
    f835m  = 'krond(a,d)*krond(b,e)*krond(c,g)*krond(f,h)'
    f836m  = 'krond(a,d)*krond(b,e)*krond(c,h)*krond(f,g)'
    f837m  = 'krond(a,d)*krond(b,f)*krond(c,e)*krond(g,h)'
    f838m  = 'krond(a,d)*krond(b,f)*krond(c,g)*krond(e,h)'
    f839m  = 'krond(a,d)*krond(b,f)*krond(c,h)*krond(e,g)'
    f840m  = 'krond(a,d)*krond(b,g)*krond(c,e)*krond(f,h)'
    f841m  = 'krond(a,d)*krond(b,g)*krond(c,f)*krond(e,h)'
    f842m  = 'krond(a,d)*krond(b,g)*krond(c,h)*krond(e,f)'
    f843m  = 'krond(a,d)*krond(b,h)*krond(c,e)*krond(f,g)'
    f844m  = 'krond(a,d)*krond(b,h)*krond(c,f)*krond(e,g)'
    f845m  = 'krond(a,d)*krond(b,h)*krond(c,g)*krond(e,f)'
    f846m  = 'krond(a,e)*krond(b,c)*krond(d,f)*krond(g,h)'
    f847m  = 'krond(a,e)*krond(b,c)*krond(d,g)*krond(f,h)'
    f848m  = 'krond(a,e)*krond(b,c)*krond(d,h)*krond(f,g)'
    f849m  = 'krond(a,e)*krond(b,d)*krond(c,f)*krond(g,h)'
    f850m  = 'krond(a,e)*krond(b,d)*krond(c,g)*krond(f,h)'
    f851m  = 'krond(a,e)*krond(b,d)*krond(c,h)*krond(f,g)'
    f852m  = 'krond(a,e)*krond(b,f)*krond(c,d)*krond(g,h)'
    f853m  = 'krond(a,e)*krond(b,f)*krond(c,g)*krond(d,h)'
    f854m  = 'krond(a,e)*krond(b,f)*krond(c,h)*krond(d,g)'
    f855m  = 'krond(a,e)*krond(b,g)*krond(c,d)*krond(f,h)'
    f856m  = 'krond(a,e)*krond(b,g)*krond(c,f)*krond(d,h)'
    f857m  = 'krond(a,e)*krond(b,g)*krond(c,h)*krond(d,f)'
    f858m  = 'krond(a,e)*krond(b,h)*krond(c,d)*krond(f,g)'
    f859m  = 'krond(a,e)*krond(b,h)*krond(c,f)*krond(d,g)'
    f860m  = 'krond(a,e)*krond(b,h)*krond(c,g)*krond(d,f)'
    f861m  = 'krond(a,f)*krond(b,c)*krond(d,e)*krond(g,h)'
    f862m  = 'krond(a,f)*krond(b,c)*krond(d,g)*krond(e,h)'
    f863m  = 'krond(a,f)*krond(b,c)*krond(d,h)*krond(e,g)'
    f864m  = 'krond(a,f)*krond(b,d)*krond(c,e)*krond(g,h)'
    f865m  = 'krond(a,f)*krond(b,d)*krond(c,g)*krond(e,h)'
    f866m  = 'krond(a,f)*krond(b,d)*krond(c,h)*krond(e,g)'
    f867m  = 'krond(a,f)*krond(b,e)*krond(c,d)*krond(g,h)'
    f868m  = 'krond(a,f)*krond(b,e)*krond(c,g)*krond(d,h)'
    f869m  = 'krond(a,f)*krond(b,e)*krond(c,h)*krond(d,g)'
    f870m  = 'krond(a,f)*krond(b,g)*krond(c,d)*krond(e,h)'
    f871m  = 'krond(a,f)*krond(b,g)*krond(c,e)*krond(d,h)'
    f872m  = 'krond(a,f)*krond(b,g)*krond(c,h)*krond(d,e)'
    f873m  = 'krond(a,f)*krond(b,h)*krond(c,d)*krond(e,g)'
    f874m  = 'krond(a,f)*krond(b,h)*krond(c,e)*krond(d,g)'
    f875m  = 'krond(a,f)*krond(b,h)*krond(c,g)*krond(d,e)'
    f876m  = 'krond(a,g)*krond(b,c)*krond(d,e)*krond(f,h)'
    f877m  = 'krond(a,g)*krond(b,c)*krond(d,f)*krond(e,h)'
    f878m  = 'krond(a,g)*krond(b,c)*krond(d,h)*krond(e,f)'
    f879m  = 'krond(a,g)*krond(b,d)*krond(c,e)*krond(f,h)'
    f880m  = 'krond(a,g)*krond(b,d)*krond(c,f)*krond(e,h)'
    f881m  = 'krond(a,g)*krond(b,d)*krond(c,h)*krond(e,f)'
    f882m  = 'krond(a,g)*krond(b,e)*krond(c,d)*krond(f,h)'
    f883m  = 'krond(a,g)*krond(b,e)*krond(c,f)*krond(d,h)'
    f884m  = 'krond(a,g)*krond(b,e)*krond(c,h)*krond(d,f)'
    f885m  = 'krond(a,g)*krond(b,f)*krond(c,d)*krond(e,h)'
    f886m  = 'krond(a,g)*krond(b,f)*krond(c,e)*krond(d,h)'
    f887m  = 'krond(a,g)*krond(b,f)*krond(c,h)*krond(d,e)'
    f888m  = 'krond(a,g)*krond(b,h)*krond(c,d)*krond(e,f)'
    f889m  = 'krond(a,g)*krond(b,h)*krond(c,e)*krond(d,f)'
    f890m  = 'krond(a,g)*krond(b,h)*krond(c,f)*krond(d,e)'
    f891m  = 'krond(a,h)*krond(b,c)*krond(d,e)*krond(f,g)'
    f892m  = 'krond(a,h)*krond(b,c)*krond(d,f)*krond(e,g)'
    f893m  = 'krond(a,h)*krond(b,c)*krond(d,g)*krond(e,f)'
    f894m  = 'krond(a,h)*krond(b,d)*krond(c,e)*krond(f,g)'
    f895m  = 'krond(a,h)*krond(b,d)*krond(c,f)*krond(e,g)'
    f896m  = 'krond(a,h)*krond(b,d)*krond(c,g)*krond(e,f)'
    f897m  = 'krond(a,h)*krond(b,e)*krond(c,d)*krond(f,g)'
    f898m  = 'krond(a,h)*krond(b,e)*krond(c,f)*krond(d,g)'
    f899m  = 'krond(a,h)*krond(b,e)*krond(c,g)*krond(d,f)'
    f8100m = 'krond(a,h)*krond(b,f)*krond(c,d)*krond(e,g)'
    f8101m = 'krond(a,h)*krond(b,f)*krond(c,e)*krond(d,g)'
    f8102m = 'krond(a,h)*krond(b,f)*krond(c,g)*krond(d,e)'
    f8103m = 'krond(a,h)*krond(b,g)*krond(c,d)*krond(e,f)'
    f8104m = 'krond(a,h)*krond(b,g)*krond(c,e)*krond(d,f)'
    f8105m = 'krond(a,h)*krond(b,g)*krond(c,f)*krond(d,e)'

    # Molecular vector of basis functions
    lmol = [ f81m,   f82m,   f83m,   f84m,   f85m,
             f86m,   f87m,   f88m,   f89m,   f810m,
             f811m,  f812m,  f813m,  f814m,  f815m,
             f816m,  f817m,  f818m,  f819m,  f820m,
             f821m,  f822m,  f823m,  f824m,  f825m,
             f826m,  f827m,  f828m,  f829m,  f830m,
             f831m,  f832m,  f833m,  f834m,  f835m,
             f836m,  f837m,  f838m,  f839m,  f840m,
             f841m,  f842m,  f843m,  f844m,  f845m,
             f846m,  f847m,  f848m,  f849m,  f850m,
             f851m,  f852m,  f853m,  f854m,  f855m,
             f856m,  f857m,  f858m,  f859m,  f860m,
             f861m,  f862m,  f863m,  f864m,  f865m,
             f866m,  f867m,  f868m,  f869m,  f870m,
             f871m,  f872m,  f873m,  f874m,  f875m,
             f876m,  f877m,  f878m,  f879m,  f880m,
             f881m,  f882m,  f883m,  f884m,  f885m,
             f886m,  f887m,  f888m,  f889m,  f890m,
             f891m,  f892m,  f893m,  f894m,  f895m,
             f896m,  f897m,  f898m,  f899m,  f8100m,
             f8101m, f8102m, f8103m, f8104m, f8105m ]

    # Temporary set for checking uniqueness
    stmp = set()
    # This set stores the elements of saaaabbcc that are redundant when 
    # we insert values of the indices.
    sintersect = set()
    # Loop through the elements of saaaabbcc
    for item in saaaabbcc:
        # Assign values to the indices
        a = item[0]
        b = item[1]
        c = item[2]
        d = item[3]
        e = item[4]
        f = item[5]
        g = item[6]
        h = item[7]
        # Temporary list for storing vectors with values
        tmp = []
        for vec in lmol:
            # Evaluate the value of the Kronecker delta products
            v = eval_krond(vec, a, b, c, d, e, f, g, h)
            tmp.append(v)
        # We need immutable objects to add in a set
        tmp = tuple(tmp)
        # Add to sintersect if the item is in stmp
        if tmp in stmp:
            sintersect.add(item)
        # Add to stmp if it isn't present
        else:
            stmp.add(tmp)
    # This function removes elements of saaaabbcc that intersect with
    # elements of sintersect.  The result is a set containing only the 
    # unique elements.
    saaaabbcc.difference_update(sintersect)

    # Print elements of saaaabbcc.
    for i in saaaabbcc:
        print(i, file=fh_2hpol)

    print('', file=fh_2hpol)
    print('~'*30, file=fh_2hpol)
    print('Number of aaaa,aaaa terms', len(saaaaaaaa), file=fh_2hpol)
    print('Number of aaaa,aabb terms', len(saaaaaabb), file=fh_2hpol)
    print('Number of aaaa,bbbb terms', len(saaaabbbb), file=fh_2hpol)
    print('Number of aaaa,bbcc terms', len(saaaabbcc), file=fh_2hpol)
    print('~'*30, file=fh_2hpol)
    print('', file=fh_2hpol)

    print('%'*30, file=fh_2hpol)
    print('Mathematica style output', file=fh_2hpol)
    print('%'*30, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Basis vectors in the experimental reference frame
    r8exp, r8mol = vectors_exp_mol(8)
    print('Experimental reference frame basis vectors', file=fh_2hpol)
    for item in r8exp:
        print(item, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Matrix for generating orientational averages
    smat, vexp, vmol = generate_smat_and_vecs(r8nn,8,False)
    print('S matrix', file=fh_2hpol)
    print(smat, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Basis vectors in the molecular reference frame
    print('Molecular reference frame basis vectors', file=fh_2hpol)
    for item in r8mol:
        print(item, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Experimental vector containing basis vectors
    print('Experimental total vector', file=fh_2hpol)
    print(vexp, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Molecular vector containing basis vectors
    print('Molecular total vector', file=fh_2hpol)
    print(vmol, file=fh_2hpol)
    print('', file=fh_2hpol)

    data, avg_gammaiiii, avg_gammaijjj = secondhpol_mathematica(saaaaaaaa, saaaaaabb, saaaabbbb, saaaabbcc)

    print('Set up molecular reference frame vectors', file=fh_2hpol)
    for item in data:
        print(item, file=fh_2hpol)
    print('', file=fh_2hpol)

    print('Second Hyperpolarizability Average Terms', file=fh_2hpol)
    print('<gamma_iiii^2> term', file=fh_2hpol)
    for item in avg_gammaiiii:
        print(item, file=fh_2hpol)
    print('', file=fh_2hpol)
    print('<gamma_ijjj^2> term', file=fh_2hpol)
    for item in avg_gammaijjj:
        print(item, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Print out the irreducible bases
    red_expbasis, red_molbasis = reduced_basis_2hpol()

    print('Irreducible experimental reference frame basis vectors', file=fh_2hpol)
    for item in red_expbasis:
        print(item, file=fh_2hpol)
    print('', file=fh_2hpol)

    print('Irreducible molecular reference frame basis vectors', file=fh_2hpol)
    for item in red_molbasis:
        print(item, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Generate the S matrix and total vectors in the irreducible bases
    smat, vexp, vmol = generate_smat_and_vecs(r8qn,8,True)
    
    # Irreducible S matrix
    print('Irreducible S matrix', file=fh_2hpol)
    print(smat, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Irreducible experimental vector containing basis vectors
    print('Irreducible experimental total vector', file=fh_2hpol)
    print(vexp, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Irreducible molecular vector containing basis vectors
    print('Irreducible molecular total vector', file=fh_2hpol)
    print(vmol, file=fh_2hpol)
    print('', file=fh_2hpol)

    # Close the files
    fh_pol.close()
    fh_hpol.close()
    fh_2hpol.close()

def fullpermutations(rank):
    '''Given a tensor of rank n, we can determine the size of the basis
       and number of linearly independent terms.  The size of the basis is:

       Nn = n! / [ 2^{n/2} * (n/2)! ]

       The number of linearly independent terms is

       Qn = \sum_{r}^{p} n! * (3*r-n+1) / [ (n-2*r)! * r! * (r+1)! ]

       Here, p is [n/2], the integer part of n/2.
    '''
    from math import factorial as fact
    from math import floor

    # Set the rank to n.
    n = rank

    # Determine the size of the basis
    Nn = fact(n) / ( pow(2,n/2) * fact(n/2) )
    Nn = int(Nn)

    # Determine the number of linearly independent terms
    p = int(floor(n/2))
    Qn = 0

    for r in range(p+1):
        Qn += fact(n) * (3*r-n+1) / ( fact(n-2*r) * fact(r) * fact(r+1) )
    Qn = int(Qn)

    return Nn, Qn

def krond(i,j):
    '''Evaluate a Kronecker delta.  Input can be any variable type, so
       use caution with this function.
    '''
    if i == j:
        return 1.0
    else:
        return 0.0

def eval_krond(string, a, b, c, d, e, f, g, h):
    '''Converts a string containing Kronecker deltas to functions that
       can be evaluated.
    '''

    # Table for replacing a-h with numbers so we can evaluate
    # Kronecker deltas
    table = { 'a' : str(a),
              'b' : str(b),
              'c' : str(c),
              'd' : str(d),
              'e' : str(e),
              'f' : str(f),
              'g' : str(g),
              'h' : str(h) }

    # Split the Kronecker delta string
    lkrond = string.split('*')

    # List for storing the values of evaluated Kronecker deltas
    l = []
    for item in lkrond:
        # Replace only the parts of the string we want
        item.replace(item[6], table[item[6]])
        item.replace(item[8], table[item[8]])
        # Evaluate the string
        tmp = eval(item)
        # Store the value of the function
        l.append(tmp)

    # Initialize the value of the Kronecker delta product
    val = 1.0

    # Evaluate the product of the Kronecker deltas.
    for item in l:
        val *= item

    return val

def alpha_convert(alpha_ab, alpha_cd, a, b, c, d):
    '''Converts a string from general molecular indices to specific 
       numerical indices.  This works for the polarizability.
    '''
    import re

    alpha_ab = re.sub(r'\[a,b\]', '[' + str(a) + ',' + str(b) + ']', alpha_ab)
    alpha_cd = re.sub(r'\[c,d\]', '[' + str(c) + ',' + str(d) + ']', alpha_cd)
    return alpha_ab, alpha_cd

def pol_mathematica(saaaa, saabb):
    '''Collects data out of the unique permutation sets for the 
       polarizability and writes it for convenient Mathematica usage. 
    '''

    # Note that a semicolon in Mathematica allows you to put multiple 
    # expressions on the same line.

    # Conversion table
    table = { 'a' : 0,
              'b' : 1,
              'c' : 2,
              'd' : 3 }

    data = []

    avg_alphaii = []
    avg_alphaij = []

    clear = 'Clear[a,b,c,d]; '
    # All indices identical (aa, aa)
    for item in saaaa:
        vmol = '; vmolaaaa = vmol;'
        data.append(clear + 'a = b = c = d = 1' + vmol)
        # Append to averaging betas
        avg_alphaii.append('vexpii.mmat.vmolaaaa;')
        avg_alphaij.append('vexpij.mmat.vmolaaaa;')
    
    tmp = {}

    # Forms with sets of non-identical indices
    for item in saabb:
        # List used for naming the vector vmol
        namelist = ['a', 'b', 'c', 'd']

        # Store the values of each item in the set
        tmp = {'a' : item[0],
               'b' : item[1],
               'c' : item[2],
               'd' : item[3] }

        # Check which variables are 1 or 2
        one = []
        two = []
        for key in tmp.keys():
            if tmp[key] == 1:
                one.append(key)
            else:
                two.append(key)

        # Strings for convenient use with Mathematica
        string_one = one[0] + ' = ' + one[1] + ' = 1'
        string_two = two[0] + ' = ' + two[1] + ' = 2'

        string = string_one + '; ' + string_two

        # Rename namelist
        name = ''
        for i in range(len(namelist)):
            namelist[i] = tmp[namelist[i]]
            if namelist[i] == 1:
                namelist[i] = 'a'
            else:
                namelist[i] = 'b'
            name = name + namelist[i]

        # Set up vmol
        vmol = '; vmol' + name + ' = vmol;'

        data.append(clear + string + vmol)
        # Append to averaging alphas
        avg_alphaii.append('vexpii.mmat.vmol' + name + ';')
        avg_alphaij.append('vexpij.mmat.vmol' + name + ';')

    return data, avg_alphaii, avg_alphaij

def hpol_mathematica(saaaaaa, saaaabb, saabbcc):
    '''Collects data out of the unique permutation sets for the first
       hyperpolarizability and writes it for convenient Mathematica 
       usage. 
    '''

    # Note that a semicolon in Mathematica allows you to put multiple 
    # expressions on the same line.

    data = []

    avg_betaiii = []
    avg_betaijj = []
    
    # Temporary lists
    aaaabbiii = []
    aaaabbijj = []
    aabbcciii = []
    aabbccijj = []

    clear = 'Clear[a,b,c,d,e,f]; '
    # All indices identical (aaa, aaa)
    for item in saaaaaa:
        vmol = '; vmolaaaaaa = vmol;'
        data.append(clear + 'a = b = c = d = e = f = 1' + vmol)
        # Append to averaging betas
        avg_betaiii.append('vexpiii.mmat.vmolaaaaaa;')
        avg_betaijj.append('vexpijj.mmat.vmolaaaaaa;')

    # Forms with sets of non-identical indices (type aaa,abb)
    tmp = {}
    for item in saaaabb:
        # List used for naming the vector vmol
        namelist = ['a', 'b', 'c', 'd', 'e', 'f']

        # Store the values of each item in the set
        tmp = {'a' : item[0],
               'b' : item[1],
               'c' : item[2],
               'd' : item[3],
               'e' : item[4],
               'f' : item[5]}

        # Check which variables are 1 or 2
        one = []
        two = []
        for key in tmp.keys():
            if tmp[key] == 1:
                one.append(key)
            else:
                two.append(key)

        # Strings for convenient use with Mathematica
        string_one = one[0] + ' = ' + one[1] + ' = ' + one[2] + ' = ' + one[3] + ' = 1'
        string_two = two[0] + ' = ' + two[1] + ' = 2'

        # Concatenate the strings
        string = string_one + '; ' + string_two

        # Rename namelist
        name = ''
        for i in range(len(namelist)):
            namelist[i] = tmp[namelist[i]]
            if namelist[i] == 1:
                namelist[i] = 'a'
            else:
                namelist[i] = 'b'
            name = name + namelist[i]

        # Set up vmol
        vmol = '; vmol' + name + ' = vmol;'

        data.append(clear + string + vmol)
        # Append to averaging betas
        #avg_betaiii.append('vexpiii.mmat.vmol' + name + ';')
        #avg_betaijj.append('vexpijj.mmat.vmol' + name + ';')
        aaaabbiii.append('vexpiii.mmat.vmol' + name + ';')
        aaaabbijj.append('vexpijj.mmat.vmol' + name + ';')

    # Forms with three sets of non-identical indices (type aab,bcc)
    tmp = {}
    for item in saabbcc:
        # List used for naming the vector vmol
        namelist = ['a', 'b', 'c', 'd', 'e', 'f']

        # Store the values of each item in the set
        tmp = {'a' : item[0],
               'b' : item[1],
               'c' : item[2],
               'd' : item[3],
               'e' : item[4],
               'f' : item[5]}

        # Check which variables are 1, 2, or 3
        one = []
        two = []
        three = []
        for key in tmp.keys():
            if tmp[key] == 1:
                one.append(key)
            elif tmp[key] == 2:
                two.append(key)
            else:
                three.append(key)

        string_one = one[0] + ' = ' + one[1] + ' = 1'
        string_two = two[0] + ' = ' + two[1] + ' = 2'
        string_three = three[0] + ' = ' + three[1] + ' = 3'

        string = string_one + '; ' + string_two + '; ' + string_three

        # Rename namelist
        name = ''
        for i in range(len(namelist)):
            namelist[i] = tmp[namelist[i]]
            if namelist[i] == 1:
                namelist[i] = 'a'
            elif namelist[i] == 2:
                namelist[i] = 'b'
            else:
                namelist[i] = 'c'
            name = name + namelist[i]

        # Set up vmol
        vmol = '; vmol' + name + ' = vmol;'

        data.append(clear + string + vmol)
        # Append to averaging betas
        #avg_betaiii.append('vexpiii.mmat.vmol' + name + ';')
        #avg_betaijj.append('vexpijj.mmat.vmol' + name + ';')
        aabbcciii.append('vexpiii.mmat.vmol' + name + ';')
        aabbccijj.append('vexpijj.mmat.vmol' + name + ';')

    # Sort the temporary lists
    aaaabbiii = sorted(aaaabbiii)
    aaaabbijj = sorted(aaaabbijj)
    aabbcciii = sorted(aabbcciii)
    aabbccijj = sorted(aabbccijj)

    # Add the temporary arrays to the final arrays.
    for k in range(len(aaaabbiii)):
        avg_betaiii.append(aaaabbiii[k]) 
        avg_betaijj.append(aaaabbijj[k]) 
    for k in range(len(aabbcciii)):
        avg_betaiii.append(aabbcciii[k]) 
        avg_betaijj.append(aabbccijj[k]) 

    return data, avg_betaiii, avg_betaijj

def secondhpol_mathematica(saaaaaaaa, saaaaaabb, saaaabbbb, saaaabbcc):
    '''Collects data out of the unique permutation sets for the second
       hyperpolarizability and writes it for convenient Mathematica 
       usage. 
    '''

    # Note that a semicolon in Mathematica allows you to put multiple 
    # expressions on the same line.

    avg_gammaiiii = []
    avg_gammaijjj = []

    data = []

    # Temporary lists
    aaaaaabbiiii = []
    aaaaaabbijjj = []
    aaaabbbbiiii = []
    aaaabbbbijjj = []
    aaaabbcciiii = []
    aaaabbccijjj = []

    clear = 'Clear[a,b,c,d,e,f,g,h]; '
    # All indices identical (aaaa, aaaa)
    for item in saaaaaaaa:
        vmol = '; vmolaaaaaaaa = vmol;'
        data.append(clear + 'a = b = c = d = e = f = g = h = 1' + vmol)
        # Append to averaging gammas
        avg_gammaiiii.append('vexpiiii.mmat.vmolaaaaaaaa;')
        avg_gammaijjj.append('vexpijjj.mmat.vmolaaaaaaaa;')

    # Forms with sets of non-identical indices (type aaaa,aabb)
    tmp = {}
    for item in saaaaaabb:
        # List used for naming the vector vmol
        namelist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        # Store all values from the list in a dictionary
        tmp = {'a' : item[0],
               'b' : item[1],
               'c' : item[2],
               'd' : item[3],
               'e' : item[4],
               'f' : item[5],
               'g' : item[6],
               'h' : item[7]}

        # Check which variables are 1 or 2
        one = []
        two = []
        for key in tmp.keys():
            if tmp[key] == 1:
                one.append(key)
            else:
                two.append(key)

        string_one = ( one[0] + ' = ' 
                     + one[1] + ' = ' 
                     + one[2] + ' = ' 
                     + one[3] + ' = '
                     + one[4] + ' = '
                     + one[5] + ' = 1' )
        string_two = two[0] + ' = ' + two[1] + ' = 2'

        string = string_one + '; ' + string_two

        # Rename namelist
        name = ''
        for i in range(len(namelist)):
            namelist[i] = tmp[namelist[i]]
            if namelist[i] == 1:
                namelist[i] = 'a'
            else:
                namelist[i] = 'b'
            name = name + namelist[i]

        # Set up vmol
        vmol = '; vmol' + name + ' = vmol;'

        data.append(clear + string + vmol)
        # Append to averaging gammas
        aaaaaabbiiii.append('vexpiiii.mmat.vmol' + name + ';')
        aaaaaabbijjj.append('vexpijjj.mmat.vmol' + name + ';')

    # Forms with sets of non-identical indices (type aaaa,bbbb)
    tmp = {}
    for item in saaaabbbb:
        # List used for naming the vector vmol
        namelist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        # Store all values from the list in a dictionary
        tmp = {'a' : item[0],
               'b' : item[1],
               'c' : item[2],
               'd' : item[3],
               'e' : item[4],
               'f' : item[5],
               'g' : item[6],
               'h' : item[7]}

        # Check which variables are 1 or 2
        one = []
        two = []
        for key in tmp.keys():
            if tmp[key] == 1:
                one.append(key)
            else:
                two.append(key)

        string_one = ( one[0] + ' = ' 
                     + one[1] + ' = '
                     + one[2] + ' = '
                     + one[3] + ' = 1' )
        string_two = ( two[0] + ' = ' 
                     + two[1] + ' = '
                     + two[2] + ' = '
                     + two[3] + ' = 2' )

        string = string_one + '; ' + string_two

        # Rename namelist
        name = ''
        for i in range(len(namelist)):
            namelist[i] = tmp[namelist[i]]
            if namelist[i] == 1:
                namelist[i] = 'a'
            else:
                namelist[i] = 'b'
            name = name + namelist[i]

        # Set up vmol
        vmol = '; vmol' + name + ' = vmol;'

        data.append(clear + string + vmol)
        # Append to averaging gammas
        aaaabbbbiiii.append('vexpiiii.mmat.vmol' + name + ';')
        aaaabbbbijjj.append('vexpijjj.mmat.vmol' + name + ';')

    # Forms with three sets of non-identical indices (type aaaa,bbcc)
    tmp = {}
    for item in saaaabbcc:
        # List used for naming the vector vmol
        namelist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

        # Store all values from the list in a dictionary
        tmp = {'a' : item[0],
               'b' : item[1],
               'c' : item[2],
               'd' : item[3],
               'e' : item[4],
               'f' : item[5],
               'g' : item[6],
               'h' : item[7]}

        # Check which variables are 1, 2, or 3
        one = []
        two = []
        three = []
        for key in tmp.keys():
            if tmp[key] == 1:
                one.append(key)
            elif tmp[key] == 2:
                two.append(key)
            else:
                three.append(key)

        string_one = ( one[0] + ' = '
                     + one[1] + ' = '
                     + one[2] + ' = '
                     + one[3] + ' = 1' )
        string_two = ( two[0] + ' = '
                     + two[1] + ' = 2' )
        string_three = ( three[0] + ' = '
                       + three[1] + ' = 3' )

        string = string_one + '; ' + string_two + '; ' + string_three

        # Rename namelist
        name = ''
        for i in range(len(namelist)):
            namelist[i] = tmp[namelist[i]]
            if namelist[i] == 1:
                namelist[i] = 'a'
            elif namelist[i] == 2:
                namelist[i] = 'b'
            else:
                namelist[i] = 'c'
            name = name + namelist[i]

        # Set up vmol
        vmol = '; vmol' + name + ' = vmol;'

        data.append(clear + string + vmol)
        # Append to averaging gammas
        aaaabbcciiii.append('vexpiiii.mmat.vmol' + name + ';')
        aaaabbccijjj.append('vexpijjj.mmat.vmol' + name + ';')

    # Add the temporary arrays to the final arrays.
    for k in range(len(aaaaaabbiiii)):
        avg_gammaiiii.append(aaaaaabbiiii[k])
        avg_gammaijjj.append(aaaaaabbijjj[k])
    for k in range(len(aaaabbbbiiii)):
        avg_gammaiiii.append(aaaabbbbiiii[k])
        avg_gammaijjj.append(aaaabbbbijjj[k])
    for k in range(len(aaaabbcciiii)):
        avg_gammaiiii.append(aaaabbcciiii[k])
        avg_gammaijjj.append(aaaabbccijjj[k])

    return data, avg_gammaiiii, avg_gammaijjj

def vectors_exp_mol(rank):
    '''Set up the vectors for Mathematica in the experimental and molecular
       frames of reference.
    '''
    import itertools
    
    # What this program does depends on the tensor rank.
    exp = []
    mol = []

    # Conversion table from the molecular to experimental reference frame
    conv = { 'a' : 'i',
             'b' : 'j',
             'c' : 'k',
             'd' : 'l',
             'e' : 'm',
             'f' : 'n',
             'g' : 'o',
             'h' : 'p' }

    if rank == 4:
        # Set up index list, set, and generator
        r4 = ['a', 'b', 'c', 'd']
        s = set()
        genl4 = itertools.permutations(r4,4)

        # Figure out all unique permutations of indices starting with 'a'
        for i in genl4:
            l1 = [i[0],i[1]]
            l2 = [i[2],i[3]]
            # Tuples for checking if a set of indices exists already.  It is
            # dumb to do this explicitly, but this is the easiest implementation.
            t1 = tuple([ l1[0], l1[1], l2[0], l2[1] ])
            t2 = tuple([ l1[0], l1[1], l2[1], l2[0] ])
            t3 = tuple([ l1[1], l1[0], l2[0], l2[1] ])
            t4 = tuple([ l1[1], l1[0], l2[1], l2[0] ])

            exists = False
            afirst = False

            if i not in s:
                if t1 in s: exists = True
                if t2 in s: exists = True
                if t3 in s: exists = True
                if t4 in s: exists = True
                if i[0] == 'a': afirst = True
                if not exists and afirst:
                    s.add(i)

        # Write the data in the lists with Mathematica style input.
        count = 0
        for item in s:
            count += 1
            # Molecular reference frame naming
            m1 = item[0]
            m2 = item[1]
            m3 = item[2]
            m4 = item[3]
            moltxt = 'f4' + str(count) + 'm = '
            moltxt += ( 'KroneckerDelta[' + m1 + ',' + m2 
                      + ']*KroneckerDelta[' + m3 + ',' + m4 + '];' )
            mol.append(moltxt)
            # Experimental reference frame naming
            e1 = conv[m1]
            e2 = conv[m2]
            e3 = conv[m3]
            e4 = conv[m4]
            exptxt = 'f4' + str(count) + ' = '
            exptxt += ( 'KroneckerDelta[' + e1 + ',' + e2 
                      + ']*KroneckerDelta[' + e3 + ',' + e4 + '];' )
            exp.append(exptxt)
    elif rank == 6:
        # Set up index list, set, and generator
        r6 = ['a', 'b', 'c', 'd', 'e', 'f']
        s = set()
        genl6 = itertools.permutations(r6,6)

        # Figure out all unique permutations of indices starting with 'a'
        for i in genl6:
            l1 = [i[0],i[1]]
            l2 = [i[2],i[3]]
            l3 = [i[4],i[5]]
            # Tuples for checking if a set of indices exists already.  It is
            # dumb to do this explicitly, but this is the easiest implementation.
            # l1, l2, l3
            t1 = tuple([ l1[0], l1[1], l2[0], l2[1], l3[0], l3[1] ])
            t2 = tuple([ l1[0], l1[1], l2[0], l2[1], l3[1], l3[0] ])
            t3 = tuple([ l1[0], l1[1], l2[1], l2[0], l3[0], l3[1] ])
            t4 = tuple([ l1[1], l1[0], l2[0], l2[1], l3[0], l3[1] ])
            t5 = tuple([ l1[0], l1[1], l2[1], l2[0], l3[1], l3[0] ])
            t6 = tuple([ l1[1], l1[0], l2[0], l2[1], l3[1], l3[0] ])
            t7 = tuple([ l1[1], l1[0], l2[1], l2[0], l3[0], l3[1] ])
            t8 = tuple([ l1[1], l1[0], l2[1], l2[0], l3[1], l3[0] ])
            # l1, l3, l2
            t9  = tuple([ l1[0], l1[1], l3[0], l3[1], l2[0], l2[1] ])
            t10 = tuple([ l1[0], l1[1], l3[0], l3[1], l2[1], l2[0] ])
            t11 = tuple([ l1[0], l1[1], l3[1], l3[0], l2[0], l2[1] ])
            t12 = tuple([ l1[1], l1[0], l3[0], l3[1], l2[0], l2[1] ])
            t13 = tuple([ l1[0], l1[1], l3[1], l3[0], l2[1], l2[0] ])
            t14 = tuple([ l1[1], l1[0], l3[0], l3[1], l2[1], l2[0] ])
            t15 = tuple([ l1[1], l1[0], l3[1], l3[0], l2[0], l2[1] ])
            t16 = tuple([ l1[1], l1[0], l3[1], l3[0], l2[1], l2[0] ])

            exists = False
            afirst = False

            if i not in s:
                # l1, l2, l3
                if t1 in s: exists = True
                if t2 in s: exists = True
                if t3 in s: exists = True
                if t4 in s: exists = True
                if t5 in s: exists = True
                if t6 in s: exists = True
                if t7 in s: exists = True
                if t8 in s: exists = True
                # l1, l3, l2
                if t9 in s: exists = True
                if t10 in s: exists = True
                if t11 in s: exists = True
                if t12 in s: exists = True
                if t13 in s: exists = True
                if t14 in s: exists = True
                if t15 in s: exists = True
                if t16 in s: exists = True
                if i[0] == 'a': afirst = True
                if not exists and afirst:
                    s.add(i)

        # Sort the set by converting to a list
        s = list(s)
        s = [b for a,b in sorted((tup[1], tup) for tup in s)]

        # Write the data in the lists with Mathematica style input.
        count = 0
        for item in s:
            count += 1
            # Molecular reference frame naming
            m1 = item[0]
            m2 = item[1]
            m3 = item[2]
            m4 = item[3]
            m5 = item[4]
            m6 = item[5]
            moltxt = 'f6' + str(count) + 'm = '
            moltxt += ( 'KroneckerDelta[' + m1 + ',' + m2
                      + ']*KroneckerDelta[' + m3 + ',' + m4  
                      + ']*KroneckerDelta[' + m5 + ',' + m6 + '];' )
            mol.append(moltxt)
            # Experimental reference frame naming
            e1 = conv[m1]
            e2 = conv[m2]
            e3 = conv[m3]
            e4 = conv[m4]
            e5 = conv[m5]
            e6 = conv[m6]
            exptxt = 'f6' + str(count) + ' = '
            exptxt += ( 'KroneckerDelta[' + e1 + ',' + e2
                      + ']*KroneckerDelta[' + e3 + ',' + e4 
                      + ']*KroneckerDelta[' + e5 + ',' + e6 + '];' )
            exp.append(exptxt)
    elif rank == 8:
        # Set up index list, set, and generator
        r8 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        s = set()
        genl8 = itertools.permutations(r8,8)
    
        for i in genl8:
            l1 = [i[0],i[1]]
            l2 = [i[2],i[3]]
            l3 = [i[4],i[5]]
            l4 = [i[6],i[7]]
            # Tuples for checking if a set of indices exists already.  It is
            # dumb to do this explicitly, but this is the easiest implementation.
            # l1, l2, l3, l4
            t1  = tuple([ l1[0], l1[1], l2[0], l2[1], l3[0], l3[1], l4[0], l4[1] ])
            t2  = tuple([ l1[0], l1[1], l2[0], l2[1], l3[0], l3[1], l4[1], l4[0] ])
            t3  = tuple([ l1[0], l1[1], l2[0], l2[1], l3[1], l3[0], l4[0], l4[1] ])
            t4  = tuple([ l1[0], l1[1], l2[1], l2[0], l3[0], l3[1], l4[0], l4[1] ])
            t5  = tuple([ l1[1], l1[0], l2[0], l2[1], l3[0], l3[1], l4[0], l4[1] ])
            t6  = tuple([ l1[0], l1[1], l2[0], l2[1], l3[1], l3[0], l4[1], l4[0] ])
            t7  = tuple([ l1[0], l1[1], l2[1], l2[0], l3[0], l3[1], l4[1], l4[0] ])
            t8  = tuple([ l1[1], l1[0], l2[0], l2[1], l3[0], l3[1], l4[1], l4[0] ])
            t9  = tuple([ l1[0], l1[1], l2[1], l2[0], l3[1], l3[0], l4[0], l4[1] ])
            t10 = tuple([ l1[1], l1[0], l2[0], l2[1], l3[1], l3[0], l4[0], l4[1] ])
            t11 = tuple([ l1[1], l1[0], l2[1], l2[0], l3[0], l3[1], l4[0], l4[1] ])
            t12 = tuple([ l1[0], l1[1], l2[1], l2[0], l3[1], l3[0], l4[1], l4[0] ])
            t13 = tuple([ l1[1], l1[0], l2[0], l2[1], l3[1], l3[0], l4[1], l4[0] ])
            t14 = tuple([ l1[1], l1[0], l2[1], l2[0], l3[0], l3[1], l4[1], l4[0] ])
            t15 = tuple([ l1[1], l1[0], l2[1], l2[0], l3[1], l3[0], l4[0], l4[1] ])
            t16 = tuple([ l1[1], l1[0], l2[1], l2[0], l3[1], l3[0], l4[1], l4[0] ])
            # l1, l3, l2, l4
            t17 = tuple([ l1[0], l1[1], l3[0], l3[1], l2[0], l2[1], l4[0], l4[1] ])
            t18 = tuple([ l1[0], l1[1], l3[0], l3[1], l2[0], l2[1], l4[1], l4[0] ])
            t19 = tuple([ l1[0], l1[1], l3[0], l3[1], l2[1], l2[0], l4[0], l4[1] ])
            t20 = tuple([ l1[0], l1[1], l3[1], l3[0], l2[0], l2[1], l4[0], l4[1] ])
            t21 = tuple([ l1[1], l1[0], l3[0], l3[1], l2[0], l2[1], l4[0], l4[1] ])
            t22 = tuple([ l1[0], l1[1], l3[0], l3[1], l2[1], l2[0], l4[1], l4[0] ])
            t23 = tuple([ l1[0], l1[1], l3[1], l3[0], l2[0], l2[1], l4[1], l4[0] ])
            t24 = tuple([ l1[1], l1[0], l3[0], l3[1], l2[0], l2[1], l4[1], l4[0] ])
            t25 = tuple([ l1[0], l1[1], l3[1], l3[0], l2[1], l2[0], l4[0], l4[1] ])
            t26 = tuple([ l1[1], l1[0], l3[0], l3[1], l2[1], l2[0], l4[0], l4[1] ])
            t27 = tuple([ l1[1], l1[0], l3[1], l3[0], l2[0], l2[1], l4[0], l4[1] ])
            t28 = tuple([ l1[0], l1[1], l3[1], l3[0], l2[1], l2[0], l4[1], l4[0] ])
            t29 = tuple([ l1[1], l1[0], l3[0], l3[1], l2[1], l2[0], l4[1], l4[0] ])
            t30 = tuple([ l1[1], l1[0], l3[1], l3[0], l2[0], l2[1], l4[1], l4[0] ])
            t31 = tuple([ l1[1], l1[0], l3[1], l3[0], l2[1], l2[0], l4[0], l4[1] ])
            t32 = tuple([ l1[1], l1[0], l3[1], l3[0], l2[1], l2[0], l4[1], l4[0] ])
            # l1, l4, l3, l2
            t33 = tuple([ l1[0], l1[1], l4[0], l4[1], l3[0], l3[1], l2[0], l2[1] ])
            t34 = tuple([ l1[0], l1[1], l4[0], l4[1], l3[0], l3[1], l2[1], l2[0] ])
            t35 = tuple([ l1[0], l1[1], l4[0], l4[1], l3[1], l3[0], l2[0], l2[1] ])
            t36 = tuple([ l1[0], l1[1], l4[1], l4[0], l3[0], l3[1], l2[0], l2[1] ])
            t37 = tuple([ l1[1], l1[0], l4[0], l4[1], l3[0], l3[1], l2[0], l2[1] ])
            t38 = tuple([ l1[0], l1[1], l4[0], l4[1], l3[1], l3[0], l2[1], l2[0] ])
            t39 = tuple([ l1[0], l1[1], l4[1], l4[0], l3[0], l3[1], l2[1], l2[0] ])
            t40 = tuple([ l1[1], l1[0], l4[0], l4[1], l3[0], l3[1], l2[1], l2[0] ])
            t41 = tuple([ l1[0], l1[1], l4[1], l4[0], l3[1], l3[0], l2[0], l2[1] ])
            t42 = tuple([ l1[1], l1[0], l4[0], l4[1], l3[1], l3[0], l2[0], l2[1] ])
            t43 = tuple([ l1[1], l1[0], l4[1], l4[0], l3[0], l3[1], l2[0], l2[1] ])
            t44 = tuple([ l1[0], l1[1], l4[1], l4[0], l3[1], l3[0], l2[1], l2[0] ])
            t45 = tuple([ l1[1], l1[0], l4[0], l4[1], l3[1], l3[0], l2[1], l2[0] ])
            t46 = tuple([ l1[1], l1[0], l4[1], l4[0], l3[0], l3[1], l2[1], l2[0] ])
            t47 = tuple([ l1[1], l1[0], l4[1], l4[0], l3[1], l3[0], l2[0], l2[1] ])
            t48 = tuple([ l1[1], l1[0], l4[1], l4[0], l3[1], l3[0], l2[1], l2[0] ])
            # l1, l2, l4, l3
            t49 = tuple([ l1[0], l1[1], l2[0], l2[1], l4[0], l4[1], l3[0], l3[1] ])
            t50 = tuple([ l1[0], l1[1], l2[0], l2[1], l4[0], l4[1], l3[1], l3[0] ])
            t51 = tuple([ l1[0], l1[1], l2[0], l2[1], l4[1], l4[0], l3[0], l3[1] ])
            t52 = tuple([ l1[0], l1[1], l2[1], l2[0], l4[0], l4[1], l3[0], l3[1] ])
            t53 = tuple([ l1[1], l1[0], l2[0], l2[1], l4[0], l4[1], l3[0], l3[1] ])
            t54 = tuple([ l1[0], l1[1], l2[0], l2[1], l4[1], l4[0], l3[1], l3[0] ])
            t55 = tuple([ l1[0], l1[1], l2[1], l2[0], l4[0], l4[1], l3[1], l3[0] ])
            t56 = tuple([ l1[1], l1[0], l2[0], l2[1], l4[0], l4[1], l3[1], l3[0] ])
            t57 = tuple([ l1[0], l1[1], l2[1], l2[0], l4[1], l4[0], l3[0], l3[1] ])
            t58 = tuple([ l1[1], l1[0], l2[0], l2[1], l4[1], l4[0], l3[0], l3[1] ])
            t59 = tuple([ l1[1], l1[0], l2[1], l2[0], l4[0], l4[1], l3[0], l3[1] ])
            t60 = tuple([ l1[0], l1[1], l2[1], l2[0], l4[1], l4[0], l3[1], l3[0] ])
            t61 = tuple([ l1[1], l1[0], l2[0], l2[1], l4[1], l4[0], l3[1], l3[0] ])
            t62 = tuple([ l1[1], l1[0], l2[1], l2[0], l4[0], l4[1], l3[1], l3[0] ])
            t63 = tuple([ l1[1], l1[0], l2[1], l2[0], l4[1], l4[0], l3[0], l3[1] ])
            t64 = tuple([ l1[1], l1[0], l2[1], l2[0], l4[1], l4[0], l3[1], l3[0] ])
            # l1, l3, l4, l2
            t65 = tuple([ l1[0], l1[1], l3[0], l3[1], l4[0], l4[1], l2[0], l2[1] ])
            t66 = tuple([ l1[0], l1[1], l3[0], l3[1], l4[0], l4[1], l2[1], l2[0] ])
            t67 = tuple([ l1[0], l1[1], l3[0], l3[1], l4[1], l4[0], l2[0], l2[1] ])
            t68 = tuple([ l1[0], l1[1], l3[1], l3[0], l4[0], l4[1], l2[0], l2[1] ])
            t69 = tuple([ l1[1], l1[0], l3[0], l3[1], l4[0], l4[1], l2[0], l2[1] ])
            t70 = tuple([ l1[0], l1[1], l3[0], l3[1], l4[1], l4[0], l2[1], l2[0] ])
            t71 = tuple([ l1[0], l1[1], l3[1], l3[0], l4[0], l4[1], l2[1], l2[0] ])
            t72 = tuple([ l1[1], l1[0], l3[0], l3[1], l4[0], l4[1], l2[1], l2[0] ])
            t73 = tuple([ l1[0], l1[1], l3[1], l3[0], l4[1], l4[0], l2[0], l2[1] ])
            t74 = tuple([ l1[1], l1[0], l3[0], l3[1], l4[1], l4[0], l2[0], l2[1] ])
            t75 = tuple([ l1[1], l1[0], l3[1], l3[0], l4[0], l4[1], l2[0], l2[1] ])
            t76 = tuple([ l1[0], l1[1], l3[1], l3[0], l4[1], l4[0], l2[1], l2[0] ])
            t77 = tuple([ l1[1], l1[0], l3[0], l3[1], l4[1], l4[0], l2[1], l2[0] ])
            t78 = tuple([ l1[1], l1[0], l3[1], l3[0], l4[0], l4[1], l2[1], l2[0] ])
            t79 = tuple([ l1[1], l1[0], l3[1], l3[0], l4[1], l4[0], l2[0], l2[1] ])
            t80 = tuple([ l1[1], l1[0], l3[1], l3[0], l4[1], l4[0], l2[1], l2[0] ])
            # l1, l4, l2, l3
            t81 = tuple([ l1[0], l1[1], l4[0], l4[1], l2[0], l2[1], l3[0], l3[1] ])
            t82 = tuple([ l1[0], l1[1], l4[0], l4[1], l2[0], l2[1], l3[1], l3[0] ])
            t83 = tuple([ l1[0], l1[1], l4[0], l4[1], l2[1], l2[0], l3[0], l3[1] ])
            t84 = tuple([ l1[0], l1[1], l4[1], l4[0], l2[0], l2[1], l3[0], l3[1] ])
            t85 = tuple([ l1[1], l1[0], l4[0], l4[1], l2[0], l2[1], l3[0], l3[1] ])
            t86 = tuple([ l1[0], l1[1], l4[0], l4[1], l2[1], l2[0], l3[1], l3[0] ])
            t87 = tuple([ l1[0], l1[1], l4[1], l4[0], l2[0], l2[1], l3[1], l3[0] ])
            t88 = tuple([ l1[1], l1[0], l4[0], l4[1], l2[0], l2[1], l3[1], l3[0] ])
            t89 = tuple([ l1[0], l1[1], l4[1], l4[0], l2[1], l2[0], l3[0], l3[1] ])
            t90 = tuple([ l1[1], l1[0], l4[0], l4[1], l2[1], l2[0], l3[0], l3[1] ])
            t91 = tuple([ l1[1], l1[0], l4[1], l4[0], l2[0], l2[1], l3[0], l3[1] ])
            t92 = tuple([ l1[0], l1[1], l4[1], l4[0], l2[1], l2[0], l3[1], l3[0] ])
            t93 = tuple([ l1[1], l1[0], l4[0], l4[1], l2[1], l2[0], l3[1], l3[0] ])
            t94 = tuple([ l1[1], l1[0], l4[1], l4[0], l2[0], l2[1], l3[1], l3[0] ])
            t95 = tuple([ l1[1], l1[0], l4[1], l4[0], l2[1], l2[0], l3[0], l3[1] ])
            t96 = tuple([ l1[1], l1[0], l4[1], l4[0], l2[1], l2[0], l3[1], l3[0] ])
    
            exists = False
            afirst = False
    
            if i not in s:
                # l1, l2, l3, l4
                if t1 in s: exists = True
                if t2 in s: exists = True
                if t3 in s: exists = True
                if t4 in s: exists = True
                if t5 in s: exists = True
                if t6 in s: exists = True
                if t7 in s: exists = True
                if t8 in s: exists = True
                if t9 in s: exists = True
                if t10 in s: exists = True
                if t11 in s: exists = True
                if t12 in s: exists = True
                if t13 in s: exists = True
                if t14 in s: exists = True
                if t15 in s: exists = True
                if t16 in s: exists = True
                # l1, l3, l2, l4
                if t17 in s: exists = True
                if t18 in s: exists = True
                if t19 in s: exists = True
                if t20 in s: exists = True
                if t21 in s: exists = True
                if t22 in s: exists = True
                if t23 in s: exists = True
                if t24 in s: exists = True
                if t25 in s: exists = True
                if t26 in s: exists = True
                if t27 in s: exists = True
                if t28 in s: exists = True
                if t29 in s: exists = True
                if t30 in s: exists = True
                if t31 in s: exists = True
                if t32 in s: exists = True
                # l1, l4, l3, l2
                if t33 in s: exists = True
                if t34 in s: exists = True
                if t35 in s: exists = True
                if t36 in s: exists = True
                if t37 in s: exists = True
                if t38 in s: exists = True
                if t39 in s: exists = True
                if t40 in s: exists = True
                if t41 in s: exists = True
                if t42 in s: exists = True
                if t43 in s: exists = True
                if t44 in s: exists = True
                if t45 in s: exists = True
                if t46 in s: exists = True
                if t47 in s: exists = True
                if t48 in s: exists = True
                # l1, l2, l4, l3
                if t49 in s: exists = True
                if t50 in s: exists = True
                if t51 in s: exists = True
                if t52 in s: exists = True
                if t53 in s: exists = True
                if t54 in s: exists = True
                if t55 in s: exists = True
                if t56 in s: exists = True
                if t57 in s: exists = True
                if t58 in s: exists = True
                if t59 in s: exists = True
                if t60 in s: exists = True
                if t61 in s: exists = True
                if t62 in s: exists = True
                if t63 in s: exists = True
                if t64 in s: exists = True
                # l1, l3, l4, l2
                if t65 in s: exists = True
                if t66 in s: exists = True
                if t67 in s: exists = True
                if t68 in s: exists = True
                if t69 in s: exists = True
                if t70 in s: exists = True
                if t71 in s: exists = True
                if t72 in s: exists = True
                if t73 in s: exists = True
                if t74 in s: exists = True
                if t75 in s: exists = True
                if t76 in s: exists = True
                if t77 in s: exists = True
                if t78 in s: exists = True
                if t79 in s: exists = True
                if t80 in s: exists = True
                # l1, l4, l2, l3
                if t81 in s: exists = True
                if t82 in s: exists = True
                if t83 in s: exists = True
                if t84 in s: exists = True
                if t85 in s: exists = True
                if t86 in s: exists = True
                if t87 in s: exists = True
                if t88 in s: exists = True
                if t89 in s: exists = True
                if t90 in s: exists = True
                if t91 in s: exists = True
                if t92 in s: exists = True
                if t93 in s: exists = True
                if t94 in s: exists = True
                if t95 in s: exists = True
                if t96 in s: exists = True
                # Make sure 'a' is the first element.
                if i[0] == 'a': afirst = True
                if not exists and afirst:
                    s.add(i)

        # Sort the set
        s = list(s)
        s = [b for a,b in sorted((tup[1], tup) for tup in s)]

        # Write the data in the lists with Mathematica style input.
        count = 0
        for item in s:
            count += 1
            # Molecular reference frame naming
            m1 = item[0]
            m2 = item[1]
            m3 = item[2]
            m4 = item[3]
            m5 = item[4]
            m6 = item[5]
            m7 = item[6]
            m8 = item[7]
            moltxt = 'f8' + str(count) + 'm = '
            moltxt += ( 'KroneckerDelta[' + m1 + ',' + m2
                      + ']*KroneckerDelta[' + m3 + ',' + m4
                      + ']*KroneckerDelta[' + m5 + ',' + m6 
                      + ']*KroneckerDelta[' + m7 + ',' + m8 + '];' )
            mol.append(moltxt)
            # Experimental reference frame naming
            e1 = conv[m1]
            e2 = conv[m2]
            e3 = conv[m3]
            e4 = conv[m4]
            e5 = conv[m5]
            e6 = conv[m6]
            e7 = conv[m7]
            e8 = conv[m8]
            exptxt = 'f8' + str(count) + ' = '
            exptxt += ( 'KroneckerDelta[' + e1 + ',' + e2
                      + ']*KroneckerDelta[' + e3 + ',' + e4
                      + ']*KroneckerDelta[' + e5 + ',' + e6 
                      + ']*KroneckerDelta[' + e7 + ',' + e8 + '];' )
            exp.append(exptxt)
    
    return exp, mol

def generate_smat_and_vecs(nvec,rank,rank8irreducible):
    '''Based on the number of basis vectors, generate the S matrix for
       determining the orientational averages.  Also make the molecular 
       and experimental reference frame vectors.
    '''

    # Dictionary of polarizability and hyperpolarizability products for
    # different averages.
    pd = { 4 : 'alpha[a,b]*alpha[c,d]',
           6 : 'beta[a,b,c]*beta[d,e,f]',
           8 : 'gamma[a,b,c,d]*gamma[e,f,g,h]' }
    
    # When the irreducible matrix for the 8th rank tensor is desired, we 
    # do something slightly different than other situations.
    if rank8irreducible:
        smat = 'rsmat = {'
        vexp = 'rvexp = {'
        vmol = 'rvmol = List/@{'

        # First generate lists containing the basis functions
        l = []
        l2 = []
        for i in range(nvec):
            txt = 'rexp' + str(rank) + str(i+1)
            l.append(txt)
            txt = 'rmol' + str(rank) + str(i+1)
            l2.append(txt)

        # Generate the irreducible S matrix
        for i in range(nvec):
            if 0 < i:
                smat += '\n'
            for j in range(nvec):
                if j == 0:
                    smat += '{' + l[i] + '*' + l[j] + ', '
                elif j == nvec - 1:
                    if i == nvec - 1:
                        smat += l[i] + '*' + l[j] + '}'
                    else:
                        smat += l[i] + '*' + l[j] + '}, '
                else:
                    smat += l[i] + '*' + l[j] + ', '
        smat += '}'

        # Generate vexp and vmol
        pol = pd[rank]
        for i in range(nvec):
            if i == 0:
                vexp += '{' + l[i] + ', '
                vmol += l2[i] + '*' + pol + ', '
            elif i == nvec - 1:
                vexp += l[i] + '}'
                vmol += l2[i] + '*' + pol
            else:
                vexp += l[i] + ', '
                vmol += l2[i] + '*' + pol + ', '
        vexp += '}'
        vmol += '}'
    else:
        smat = 'smat = {'
        vexp = 'vexp = {'
        vmol = 'vmol = List/@{'

        # First generate a list containing the basis functions
        l = []
        for i in range(nvec):
            txt = 'f' + str(rank) + str(i+1)
            l.append(txt)
        
        # Generate smat
        for i in range(nvec):
            if 0 < i:
                smat += '\n'
            for j in range(nvec):
                if j == 0:
                    smat += '{' + l[i] + '*' + l[j] + ', '
                elif j == nvec - 1:
                    if i == nvec - 1:
                        smat += l[i] + '*' + l[j] + '}'
                    else:
                        smat += l[i] + '*' + l[j] + '}, '
                else:
                    smat += l[i] + '*' + l[j] + ', '
        smat += '}'

        # Generate vexp and vmol
        pol = pd[rank]
        for i in range(nvec):
            if i == 0:
                vexp += '{' + l[i] + ', '
                vmol += l[i] + 'm' + '*' + pol + ', '
            elif i == nvec - 1:
                vexp += l[i] + '}'
                vmol += l[i] + 'm' + '*' + pol
            else:
                vexp += l[i] + ', '
                vmol += l[i] + 'm' + '*' + pol + ', '
        vexp += '}'
        vmol += '}'

    return smat, vexp, vmol

def reduced_basis_2hpol():
    '''Store the irreducible basis for the experimental and molecular reference frames
       for the 8th rank tensor average.
    '''

    # This is the irreducible basis set for the 8th rank tensor average.
    reduced_expbasis = ['f81 - f8104 - f8105 - f889 - f890', 
                        '-f8101 - f8102 + f82 - f874 - f875', 
                        'f8101 + f8102 + f8104 + f8105 + f83 + f874 + f875 + f889 + f890', 
                        'f8104 + f84 + f889', 
                        'f8101 + f85 + f874', 
                        '-f8101 - f8104 + f86 - f874 - f889', 
                        'f8105 + f87 + f890', 
                        '-f8101 - f8102 - f8104 - f8105 - f860 - f874 - f875 + f88 - f889 - f890 - f899', 
                        'f8101 + f8102 + f8104 + f860 + f874 + f875 + f889 + f89 + f899', 
                        'f810 + f8102 + f875', 
                        'f811 + f860 + f899', 
                        '-f8102 + f812 - f860 - f875 - f899', 
                        '-f8102 - f8105 + f813 - f875 - f890', 
                        '-f8104 + f814 - f860 - f889 - f899', 
                        'f8102 + f8104 + f8105 + f815 + f860 + f875 + f889 + f890 + f899', 
                        '-f8102 + f816 - f884 - f887 - f899', 
                        'f8101 + 2*f8102 + f8104 + f8105 + f817 - f869 + f875 + f887 + f890 + f899', 
                        '-f8101 - f8102 - f8104 - f8105 + f818 + f869 - f875 + f884 - f890', 
                        'f819 + f884 + f899', 
                        '-f8101 - f8102 - f8104 - f8105 + f820 + f869 - f899', 
                        'f8101 + f8102 + f8104 + f8105 + f821 - f869 - f884', 
                        'f8102 + f822 + f887', 
                        'f8101 + f8102 + f8104 + f8105 + f823 + f860 - f869 + f875 + f889 + f890 + f899', 
                        '-f8101 - 2*f8102 - f8104 - f8105 + f824 - f860 + f869 - f875 - f887 - f889 - f890 - f899', 
                        '-f8102 + f825 - f875 - f887 - f890', 
                        'f826 - f860 - f884 - f889 - f899', 
                        'f8102 + f827 + f860 + f875 + f884 + f887 + f889 + f890 + f899', 
                        'f828 + f875 + f890', 
                        'f829 + f860 + f889', 
                        'f830 - f860 - f875 - f889 - f890', 
                        'f8102 + f8104 + f8105 + f831 + f884 + f887 + f889 + f890 + f899', 
                        '-f8102 - f8104 - f8105 + f832 + f869 + f874 - f887 - f890 - f899', 
                        'f833 - f869 - f874 - f884 - f889', 
                        '-f8101 - f8102 - f8104 - f8105 + f834 - f884 - f886 - f887 - f889 - f890 - f899', 
                        'f8101 + f8102 + f8104 + f8105 + f835 - f869 + f886 + f887 + f889 + f890 + f899', 
                        'f836 + f869 + f884', 
                        'f8101 + f837 + f886', 
                        '-f8101 - f8102 - f8104 - f8105 + f838 - f860 + f869 - f875 - f886 - f887 - f889 - f890 - f899', 
                        'f8102 + f8104 + f8105 + f839 + f860 - f869 + f875 + f887 + f889 + f890 + f899', 
                        '-f8101 + f840 - f874 - f886 - f889', 
                        'f8101 + f8102 + f8104 + f8105 + f841 + f860 + f874 + f875 + f884 + f886 + f887 + 2*f889 + f890 + f899', 
                        '-f8102 - f8104 - f8105 + f842 - f860 - f875 - f884 - f887 - f889 - f890 - f899', 
                        ' f843 + f874 + f889', 
                        'f844 - f860 - f874 - f875 - f889', 
                        'f845 + f860 + f875', 
                        '-f8104 + f846 - f884 - f889 - f899', 
                        'f8102 + f8104 + f8105 + f847 - f869 - f874 + f899', 
                        '-f8102 - f8105 + f848 + f869 + f874 + f884 + f889', 
                        'f8101 + f8102 + f8104 + f849 + f884 + f886 + f887 + f889 + f899', 
                        '-f8101 - 2*f8102 - f8104 - f8105 + f850 + f869 - f875 - f886 -f887 - f889 - f890 - f899', 
                        'f8102 + f8105 + f851 - f869 + f875 - f884 + f890', 
                        '-f8101 - f8102 + f852 - f886 - f887', 
                        'f8101 + 2*f8102 + f8104 + f8105 + f853 + f860 - f869 + f875 + f886 + f887 + f889 + f890 + f899', 
                        '-f8102 - f8104 - f8105 + f854 - f860 + f869 - f875 - f889 - f890 - f899', 
                        'f8101 + f8102 + f855 + f874 + f875 + f886 + f887 + f889 + f890', 
                        '-f8101 - f8102 - f8104 + f856 - f860 - f874 - f875 - f884 - f886 - f887 - 2*f889 - f890 - f899', 
                        'f8104 + f857 + f860 + f884 + f889 + f899', 
                        'f858 - f874 - f875 - f889 - f890', 
                        'f859 + f860 + f874 + f875 + f889 + f890', 
                        '-f8102 - f8105 + f861 - f887 - f890', 
                        'f862 + f869 + f874', 
                        'f8102 + f8105 + f863 - f869 - f874 + f887 + f890', 
                        '-f8101 - f8104 + f864 - f886 - f889', 
                        'f8101 + f8102 + f8104 + f8105 + f865 - f869 + f875 + f886 + f887 + f889 + f890', 
                        '-f8102 - f8105 + f866 + f869 - f875 - f887 - f890', 
                        'f8101 + f8102 + f8104 + f8105 + f867 + f886 + f887 + f889 + f890', 
                        '-f8101 - f8102 - f8104 - f8105 + f868 + f869 - f886 - f887 - f889 - f890', 
                        '-f8101 - f8102 - f8104 - f8105 + f870 - f874 - f875 - f886 - f887 - f889 - f890', 
                        'f8101 + f8104 + f871 + f874 + f886 + f889', 
                        'f8102 + f8105 + f872 + f875 + f887 + f890', 
                        'f873 + f874 + f875', 
                        'f876 + f887 + f890', 
                        'f877 + f884 + f889', 
                        'f878 - f884 - f887 - f889 - f890',
                        'f879 + f886 + f889', 
                        'f880 - f884 - f886 - f887 - f889', 
                        'f881 + f884 + f887', 
                        'f882 - f886 - f887 - f889 - f890', 
                        'f883 + f884 + f886 + f887 + f889 + f890', 
                        'f885 + f886 + f887', 
                        'f888 + f889 + f890', 
                        'f8102 + f8105 + f891', 
                        'f8104 + f892 + f899', 
                        '-f8102 - f8104 - f8105 + f893 - f899', 
                        'f8101 + f8104 + f894', 
                        '-f8101 - f8102 - f8104 + f895 - f899', 
                        'f8102 + f896 + f899', 
                        '-f8101 - f8102 - f8104 - f8105 + f897', 
                        'f8101 + f8102 + f8104 + f8105 + f898 + f899', 
                        'f8100 + f8101 + f8102', 
                        'f8103 + f8104 + f8105']

    # Construct the reduced basis for the molecular reference frame.
    reduced_molbasis = [] 
    count = 1
    for item in reduced_expbasis:
        tmp = item.split()
        for i in range(0,len(tmp),2):
            tmp[i] += 'm'
        for i in range(1,len(tmp),2):
            tmp[i] = ' ' + tmp[i] + ' '
        txt = 'rmol8' + str(count) + ' = '
        for i in range(len(tmp)):
            txt += tmp[i]
        txt += ';'
        reduced_molbasis.append(txt)
        count += 1

    # Modify the reduced basis for the experimental reference frame.
    count = 1
    for item in reduced_expbasis:
        txt = 'rexp8' + str(count) + ' = '
        txt = txt + item + ';'
        reduced_expbasis[count-1] = txt
        count += 1

    return reduced_expbasis, reduced_molbasis

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
