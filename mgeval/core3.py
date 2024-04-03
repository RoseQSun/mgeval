# coding:utf-8
"""core.py
Include feature extractor and musically informed objective measures.
"""
import pretty_midi
import numpy as np
import math


# feature extractor
def extract_feature(_file):
    """
    This function extracts two midi feature:
    pretty_midi object: https://github.com/craffel/pretty-midi
    midi_pattern: https://github.com/vishnubob/python-midi

    Returns:
        dict(pretty_midi: pretty_midi object,
             midi_pattern: midi pattern contains a list of tracks)
    """
    feature = {'pretty_midi': pretty_midi.PrettyMIDI(_file)}
    return feature


# musically informed objective measures.
class metrics(object):
    def total_used_pitch(self, feature):
        """
        total_used_pitch (Pitch count): The number of different pitches within a sample.

        Returns:
        'used_pitch': pitch count, scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        sum_notes = np.sum(piano_roll, axis=1)
        used_pitch = np.sum(sum_notes > 0)
        return used_pitch

    def total_pitch_class_histogram(self, feature):
        """
        total_pitch_class_histogram (Pitch class histogram):
        The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
        In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

        Returns:
        'histogram': histrogram of 12 pitch, with weighted duration shape 12
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        histogram = np.zeros(12)
        for i in range(0, 128):
            pitch_class = i % 12
            histogram[pitch_class] += np.sum(piano_roll, axis=1)[i]
        histogram = histogram / sum(histogram)
        return histogram

    def bar_pitch_class_histogram(self, feature, track_num=1, num_bar=None, bpm=120):
        """
        bar_pitch_class_histogram (Pitch class histogram per bar):

        Args:
        'bpm' : specify the assigned speed in bpm, default is 120 bpm.
        'num_bar': specify the number of bars in the midi pattern, if set as None, round to the number of complete bar.
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'histogram': with shape of [num_bar, 12]
        """

        # todo: deal with more than one time signature cases
        pm_object = feature['pretty_midi']
        if num_bar is None:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_length = 60. / bpm * numer * 4 / deno * 100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll, (1, 0))
            actual_bar = len(piano_roll) / bar_length
            num_bar = int(round(actual_bar))
            bar_length = int(round(bar_length))
        else:
            numer = pm_object.time_signature_changes[-1].numerator
            deno = pm_object.time_signature_changes[-1].denominator
            bar_length = 60. / bpm * numer * 4 / deno * 100
            piano_roll = pm_object.instruments[track_num].get_piano_roll(fs=100)
            piano_roll = np.transpose(piano_roll, (1, 0))
            actual_bar = len(piano_roll) / bar_length
            bar_length = int(math.ceil(bar_length))

        if actual_bar > num_bar:
            mod = np.mod(len(piano_roll), bar_length*128)
            piano_roll = piano_roll[:-np.mod(len(piano_roll), bar_length)].reshape((num_bar, -1, 128))  # make exact bar
        elif actual_bar == num_bar:
            piano_roll = piano_roll.reshape((num_bar, -1, 128))
        else:
            piano_roll = np.pad(piano_roll, ((0, int(num_bar * bar_length - len(piano_roll))), (0, 0)), mode='constant', constant_values=0)
            piano_roll = piano_roll.reshape((num_bar, -1, 128))

        bar_histogram = np.zeros((num_bar, 12))
        for i in range(0, num_bar):
            histogram = np.zeros(12)
            for j in range(0, 128):
                pitch_class = j % 12
                histogram[pitch_class] += np.sum(piano_roll[i], axis=0)[j]
            if sum(histogram) != 0:
                bar_histogram[i] = histogram / sum(histogram)
            else:
                bar_histogram[i] = np.zeros(12)
        return bar_histogram

    def pitch_class_transition_matrix(self, feature, normalize=0):
        """
        pitch_class_transition_matrix (Pitch class transition matrix):
        The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
        The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.

        Args:
        'normalize' : If set to 0, return transition without normalization.
                      If set to 1, normalizae by row.
                      If set to 2, normalize by entire matrix sum.
        Returns:
        'transition_matrix': shape of [12, 12], transition_matrix of 12 x 12.
        """
        pm_object = feature['pretty_midi']
        transition_matrix = pm_object.get_pitch_class_transition_matrix()

        if normalize == 0:
            return transition_matrix

        elif normalize == 1:
            sums = np.sum(transition_matrix, axis=1)
            sums[sums == 0] = 1
            return transition_matrix / sums.reshape(-1, 1)

        elif normalize == 2:
            return transition_matrix / sum(sum(transition_matrix))

        else:
            print("invalid normalization mode, return unnormalized matrix")
            return transition_matrix

    def pitch_range(self, feature):
        """
        pitch_range (Pitch range):
        The pitch range is calculated by subtraction of the highest and lowest used pitch in semitones.

        Returns:
        'p_range': a scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        pitch_index = np.where(np.sum(piano_roll, axis=1) > 0)
        p_range = np.max(pitch_index) - np.min(pitch_index)
        return p_range

    def avg_IOI(self, feature):
        """
        avg_IOI (Average inter-onset-interval):
        To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

        Returns:
        'avg_ioi': a scalar for each sample.
        """

        pm_object = feature['pretty_midi']
        onset = pm_object.get_onsets()
        ioi = np.diff(onset)
        avg_ioi = np.mean(ioi)
        return avg_ioi

    # def chord_dependency(self, feature, bar_chord, bpm=120, num_bar=None, track_num=1):
    #     pm_object = feature['pretty_midi']
    #     # compare bar chroma with chord chroma. calculate the ecludian
    #     bar_pitch_class_histogram = self.bar_pitch_class_histogram(pm_object, bpm=bpm, num_bar=num_bar, track_num=track_num)
    #     dist = np.zeros((len(bar_pitch_class_histogram)))
    #     for i in range((len(bar_pitch_class_histogram))):
    #         dist[i] = np.linalg.norm(bar_pitch_class_histogram[i] - bar_chord[i])
    #     average_dist = np.mean(dist)
    #     return average_dist
