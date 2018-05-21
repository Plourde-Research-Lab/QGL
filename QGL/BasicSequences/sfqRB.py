from ..PulsePrimitives import *
from ..Compiler import compile_to_hardware
from ..PulseSequencePlotter import plot_pulse_files
from ..Cliffords import clifford_seq, clifford_mat, inverse_clifford
from .helpers import create_cal_seqs

import os
from csv import reader
import numpy as np
from functools import reduce


def create_sfq_RB_seqs(numQubits, lengths, repeats=32, interleaveGate=None, recovery=True):
    """Create a list of lists of Clifford gates to implement RB. """
    if numQubits == 1:
        cliffGroupSize = 24
    elif numQubits == 2:
        cliffGroupSize = 11520
    else:
        raise Exception("Can only handle one or two qubits.")

    #Create lists of of random integers
    #Subtract one from length for recovery gate
    seqs = []
    for length in lengths:
        seqs += np.random.randint(0, cliffGroupSize,
                                  size=(repeats, length - 1)).tolist()

    #Possibly inject the interleaved gate
    if interleaveGate:
        newSeqs = []
        for seq in seqs:
            newSeqs.append(np.vstack((np.array(
                seq, dtype=np.int), interleaveGate * np.ones(
                    len(seq), dtype=np.int))).flatten(order='F').tolist())
        seqs = newSeqs

    if recovery:
        #Calculate the recovery gate
        for seq in seqs:
            if len(seq) == 1:
                mat = clifford_mat(seq[0], numQubits)
            else:
                mat = reduce(lambda x, y: np.dot(y, x),
                             [clifford_mat(c, numQubits) for c in seq])
            seq.append(inverse_clifford(mat))

    return seqs

def SFQQubitRB(qubit, seqs, purity=False, showPlot=False):
    """Single qubit randomized benchmarking using 90 and 180 generators.

    Parameters
    ----------
    qubit : logical channel to implement sequence (LogicalChannel)
    seqs : list of lists of Clifford group integers
    showPlot : whether to plot (boolean)
    """

    seqsBis = []
    op = [Id(qubit, length=0), Y90m(qubit), X90(qubit)]
    for ct in range(3 if purity else 1):
        for seq in seqs:
            seqsBis.append(reduce(operator.add, [clifford_seq(c, qubit)
                                                for c in seq]))
            #append tomography pulse to measure purity
            seqsBis[-1].append(op[ct])
            #append measurement
            seqsBis[-1].append(MEAS(qubit))

    #Tack on the calibration sequences
    seqsBis += create_cal_seqs((q1, ), 2)

    metafile = compile_to_hardware(seqsBis, 'RB/RB')

    if showPlot:
        plot_pulse_files(metafile)
    return metafile



    #Hack for limited APS waveform memory and break it up into multiple files
    #We've shuffled the sequences so that we loop through each gate length on the inner loop
    numRandomizations = 36
    for ct in range(numRandomizations):
        chunk = seqs[ct::numRandomizations]
        chunk1 = chunk[::2]
        chunk2 = chunk[1::2]
        #Tack on the calibration scalings
        chunk1 += [[Id(qubit), measBlock], [X(qubit), measBlock]]
        metafile = compile_to_hardware(chunk1,
                                        'RB/RB',
                                        suffix='_{0}'.format(2 * ct + 1))
        chunk2 += [[Id(qubit), measBlock], [X(qubit), measBlock]]
        metafile = compile_to_hardware(chunk2,
                                        'RB/RB',
                                        suffix='_{0}'.format(2 * ct + 2))

    if showPlot:
        plot_pulse_files(metafile)
    return metafile
