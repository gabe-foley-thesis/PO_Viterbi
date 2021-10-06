import subprocess
import pov.align as align
import pov.parameters as params
import pov.utilities as utilities
import os
import pickle
import timeit
import csv
from datetime import datetime
import itertools
import pov.alignment_profile as aln_profile
import pov.sub_matrix as sub_matrix
import pov.sequence
from pov.sym import Alphabet
from collections import defaultdict

aa_skip = "BJOUZ"

change_params = {'tau': 0.000000000001, 'epsilon': 0.003, 'delta': 0.000004, 'emissionX':
    0.5,
                 'emissionY':
                     0.5}


change_params = {'tau': 0.0002, 'epsilon': 0.04, 'delta': 0.0002, 'emissionX': 0.2, 'emissionY':
    0.2}

log_transform = True



Protein_Alphabet_wB_X_Z = Alphabet('ABCDEFGHIKLMNPQRSTVWYXZ')

def run_qscore(name, aln_type, parameters, specific_files=None, save=False, outpath=""):
    base_dir = "./bench1.0/" +  name

    in_dir = base_dir + "/in/"
    ref_dir = base_dir + "/ref/"
    out_dir = "./qscore_alignments/" + aln_type + "_" + name

    qscore_dict = defaultdict(dict)


    files = os.listdir(in_dir)

    file_count = 0

    start_time = timeit.default_timer()

    now = datetime.now()

    dt_string = now.strftime("%Y/%m/%d_%H:%M")

    # Add trailing slash to output directory if it isn't there
    outpath = outpath + "/" if outpath[-1] != "/" else outpath

    param_name = f"t={parameters['tau']}e={parameters['epsilon']}d={parameters['delta']}x={parameters['emissionX']}y={parameters['emissionY']}"

    output_file = "./qscore_alignments/" + aln_type + "_" + name + param_name + ".csv"


    if os.path.exists(outpath + name + ".p"):
        curr_dict = pickle.load(open(outpath + name + ".p", "rb"))
    else:
        curr_dict = {param_name : {}}

    if os.path.exists(outpath + name + "_best.p"):
        best_dict = pickle.load(open(outpath + name + "_best.p", "rb"))
    else:
        best_dict = {}

    if os.path.exists(outpath + "time.p"):
        time_dict = pickle.load(open(outpath + "time.p", "rb"))
    else:
        time_dict = {}

    failures = []

    with open (output_file, 'w+') as output:


        writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Tool', 'Dataset', 'Name', 'Q', 'TC', 'M', 'C'])

        # If we don't already have a directory created to save the alignments, lets make one
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        for file in files:

            failed = False


            if file != ".DS_Store":

                seqs = sequence.readFastaFile(in_dir + file, alphabet=Protein_Alphabet_wB_X_Z)

                for seq in seqs:
                    if any(skip in seq.sequence for skip in aa_skip):
                        print("failed on " + seq.name)
                        failures.append(file)
                        failed = True

                if not failed:


                    qscore_dict[file] = defaultdict(dict)

                    if not specific_files  or file in specific_files:

                        if param_name not in curr_dict:
                            curr_dict[param_name] = {}

                        # print (curr_dict)
                        file_count +=1

                        single_time = timeit.default_timer()

                        print (file)


                        # change_params = {'tau': 0.000002, 'epsilon': 0.0001, 'delta': 0.0002, 'emissionX': 0.2, 'emissionY':
                        #     0.2}
                        # change_params = {'tau': 0.00000000002, 'epsilon': 0.000175, 'delta': 0.00031, 'emissionX':
                        #     0.002,
                        #                  'emissionY':
                        #     0.002}
                        #
                        # change_params = {'tau': 0.1, 'epsilon': 0.02, 'delta': 0.01, 'emissionX':
                        #     0.5,
                        #                  'emissionY':
                        #     0.5}
                        # Update parameters using Baum Welch
                        for seq_order in list(itertools.combinations(seqs, 2)):
                            profiles = [aln_profile.AlignmentProfile([x]) for x in seq_order]


                            # change_params = bw.runBaumWelch(parameters, profiles, aln_type)


                        print (parameters)
                        # print (change_params)

                        aligned_profile = align.align_seqs(in_dir + file, out_dir + "/" + file + ".aln", aln_type=aln_type,
                                                           params=parameters, subsmat=sub_matrix.blosum62EstimatedWithX_dict,
                                                           log_transform=log_transform)


                        process = subprocess.Popen("qscore -test %s -ref %s -cline -modeler" % (out_dir + "/" + file + ".aln",
                                                                                                ref_dir + file),
                                                   stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

                        out = process.communicate()[0]
                        errcode = process.returncode

                        print ('running')
                        print (errcode)

                        scores = [x.strip() for x in out.decode('utf-8').split(";")[2:]]

                        # scores = [x.split("=")[1] for x in scores]

                        # print (aligned_profile)
                        print (file)

                        print('\nScores be')
                        print(scores)

                        for score in scores:
                            score_type = score.split("=")[0].strip()
                            score_value = score.split("=")[1].strip()
                            qscore_dict[file][score_type] = score_value

                        curr_dict[param_name][file] = (scores, aligned_profile)

                        update_best_dict(best_dict, file, scores, param_name)

                        if scores and "=" in scores[0]:
                            writer.writerow([aln_type + "_" + param_name + "_log=" + str(log_transform), name, file,
                                             scores[0].split(
                                "=")[1],
                                             scores[1].split(
                                "=")[1],
                                             scores[2].split("=")[1], scores[3].split("=")[1]])

                        else:
                            failures.append(file)

                        # if file not in curr_dict[param_name].keys():
                        #     curr_dict[param_name][file] = (scores, aligned_profile)
                        # else:
                        #     curr_dict[param_name][file] = (scores, aligned_profile)
                        #

                        total_seconds = timeit.default_timer() - start_time
                        single_seconds = timeit.default_timer() - single_time

                        if save:

                            pickle.dump(curr_dict, open(outpath + aln_type + "_" + name + ".p", "wb"))
                            pickle.dump(best_dict, open(outpath + aln_type + "_" + name + "_best.p", "wb"))

                    if save:

                        if name in time_dict:
                            if total_seconds < time_dict[name][0]:
                                time_dict[name] = (total_seconds, dt_string)
                                print("New best time - " + utilities.format_time(total_seconds))
                        else:
                            time_dict[name] = (total_seconds, dt_string)
                            print("New best time - " + utilities.format_time(total_seconds))

                    pickle.dump(time_dict, open(outpath + aln_type + "_" + "time.p", "wb"))
    print ('These files failed ' )
    print (failures)
    return qscore_dict



def update_best_dict(best_dict, file, curr_scores, param_name):


    for score in curr_scores:
        name = score.split("=")[0]
        curr_score = score.split("=")[1]
        if file not in best_dict.keys():

            best_dict[file] = {name : (float(curr_score), param_name)}
            print ("New best score for " + name + " of " + curr_score)

        else:
            if name not in best_dict[file]:
                best_dict[file][name] = (float(curr_score), param_name)
                print ("New best score for " + name + " of " + curr_score)


            elif float(curr_score) > best_dict[file][name][0]:
                best_dict[file][name] = (float(curr_score), param_name)
                print ("New best score for " + name + " of " + curr_score)



def load_best_dict(in_dir, filenames = []):

    if not filenames:
        files = os.listdir(in_dir)
        for name in files:
            if "_best" in name:
                filenames.append(name)

    for name in filenames:
        best_dict = pickle.load(open(in_dir + name, 'rb'))

        for k,v in best_dict.items():
            print (k)
            for v1, v2 in v.items():
                print (v1, v2)

def load_time_dict(in_dir):
    time_dict = pickle.load(open(in_dir + "time.p", "rb"))

    for k, v in time_dict.items():
        print(k, v)



benchmark_names_dna = ['bali2dna', 'bali2dnaf']

benchmark_names_test_short = ['ox', 'oxm', 'oxx', 'prefab4', 'prefab4ref',
                           'prefab4refm', 'sabre', 'sabrem', 'bali3', 'bali3pdb', 'bali3pdm',]


benchmark_names_test_short = ['bali3pdb', 'bali3pdm', 'prefab4', 'prefab4ref',
                           'prefab4refm', 'oxx', 'ox', 'oxm', 'sabrem', 'bali3']

benchmark_names_test_short = ['bali3_single']




# benchmark_names_all = benchmark_names_dna + benchmark_names_protein

# benchmark_names_test = ['bali3_test', 'bali3pdb_test', 'bali3pdm_test', 'ox_test', 'oxm_test', 'oxx_test', 'prefab4_test', 'prefab4ref_test',
#                            'prefab4refm_test', 'sabre_test', 'sabrem_test']

# 'oxx_test', 'prefab4_test'

aln_types = ['poviterbi']

# aln_types = ['mea', 'pomea', 'viterbi', 'poviterbi']


# benchmark_names_test_short = ['prefab4ref_test_single']
# benchmark_names_test_short = ['bali3_test', 'ox_test', 'prefab4ref_test', 'prefab4refm_test', 'sabre_test', 'sabrem_test']
# benchmark_names_test_short = ['prefab4ref','bali3', 'bali3pdb', 'bali3pdm', 'ox', 'oxm', 'oxx', 'prefab4',
#                            'prefab4refm', 'sabre', 'sabrem']
# 'prefab4ref','bali3', 'bali3pdb', 'bali3pdm' didn't do bali3pdm

# bali3pdm, ox, oxm, oxx, prefab4 not done for both

# benchmark_names_test_short = ['prefab4refm']

benchmark_names_test_short = ['sabre']
# benchmark_names_test_short = ['ox_test', 'sabre_test']


# benchmark_names_test_short = ['bali3']
# for name in benchmark_names_test_short:
#     for aln_type in aln_types:
#         run_qscore(name, aln_type=aln_type, parameters=params.test_params3, specific_files=['581t17' ],
#
#                    save=True, \
#                                                                                      outpath='./pickle_files/test')



def get_qscores(name):
    base_dir = "./bench1.0/" +  name

    in_dir = base_dir + "/in/"
    qscores = base_dir + "/qscore/"

    methods = os.listdir(qscores)
    tests = os.listdir(in_dir)

    qscore_dict = defaultdict(dict)

    for method in methods:
        qscore_dict[method] = defaultdict(dict)

        for line in open(qscores + method):
            test_from_line = line.split("ref/")[1].split(";")[0]


            if test_from_line in tests:
                qscore_dict[method][test_from_line] = defaultdict(dict)
                results = line.split("ref/")[1].split(";")[1:]
                for result in results:
                    result_type = result.split("=")[0].strip()
                    result_value = result.split("=")[1].strip()
                    qscore_dict[method][test_from_line][result_type] = result_value

    return qscore_dict

def make_graphs(result_dict, *metrics):

    for metric in metrics:
        print (metric)


epsilons = [0.006, 0.007, 0.009, 0.1, 0.0001, 0.03]
deltas = [0.002, 0.005, 0.0005, 0.4, 0.003, 0.04]

for e in epsilons:
    for d in deltas:

        for name in benchmark_names_test_short:

            change_params = {'tau': 0.002, 'epsilon': e, 'delta': d, 'emissionX': 0.88, 'emissionY':
                0.88}

            # other_methods_dict = get_qscores(name)

            qscore_dict = defaultdict(dict)
            for aln_type in aln_types:
                qscores = run_qscore(name, aln_type=aln_type, parameters=change_params, specific_files=None, save=True,
                                     outpath='./pickle_files/test')

            qscore_dict[aln_type] = qscores
            print (qscore_dict)
