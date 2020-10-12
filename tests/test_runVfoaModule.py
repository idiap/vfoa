import numpy as np

try:
    from commonTasks.evaluation.classification_errorMeasures import FrameBasedEvaluator
    FRAME_BASED_EVALUATOR_AVAILABLE = True
except ImportError:
    FRAME_BASED_EVALUATOR_AVAILABLE = False

from vfoaModule.vfoa.vfoa_module import VFOAModule, Person, Target

LABEL2CODE = {'unfocused': 0, 'aversion': 0, 'robot': 1, 'camera': 1, 'person': 2, 'person 1': 2, 'person 2': 2}


def main(inputDataFile, annotations=None, model=None):

    data = np.loadtxt(inputDataFile, dtype=str, delimiter=',')

    # Model
    if model is None:
        model = ['gaussianModel', 'geometricalModel', 'gazeProbability', 'HMM'][3]

    vfoaModule = VFOAModule(model)
    if model == 'gaussianModel':
        # var_aversion, var_target, var_noise, prob_aversion
        vfoaModule.set_model_parameters([400. * np.eye(2), 100. * np.eye(2), 0. * np.eye(2), 0.0001])
    elif model == 'geometricalModel':
        # threshold
        vfoaModule.set_model_parameters([45])
    elif model == 'gazeProbability':
        # var_aversion, var_target, var_noise, prob_aversion
        vfoaModule.set_model_parameters([400. * np.eye(2), None, None, 0.0001])
    elif model == 'HMM':
        # headpose_history_duration, headpose_ref_duration, headpose_ref_gap, headpose_prev_duration, headpose_prev_gap,
        # headgaze_ref_factor, headgaze_prev_factor, prior_min, prior_std, prior_unfocused, default_prob_ii
        parameterList = [None] * 11
        parameterList[0] = 35
        parameterList[5] = np.array([.4, .6])
        parameterList[6] = np.array([0., 0.])
        parameterList[7] = .001
        parameterList[8] = np.array([12 * np.pi / 180, 13 * np.pi / 180])
        parameterList[9] = .196
        vfoaModule.set_model_parameters(parameterList)

    # Ground truth
    groundTruth = np.load(annotations) if annotations is not None else None

    # Evaluation
    nb_correct, nb_total = np.zeros(2), np.zeros(2)
    nb_correct_gt, nb_total_gt = np.zeros(2), np.zeros(2)

    estimationList, baselineList, groundTruthList = [], [], []

    timestamp_current = int(data[0, 0])
    frameData, frameIndex = [], 0
    for dataline in data:
        if int(dataline[0]) != timestamp_current:
            frameIndex += 1

            # Process each person
            personDict, targetDict = {}, {}
            for tracklet in frameData:
                # personDict, targetDict = {}, {}

                # Get targets
                targetDict['Robot'] = Target('Robot', [0, 0, 0])
                for tracklet2 in frameData:
                    if tracklet2[1] != tracklet[1]:
                        name = 'Person {}'.format(tracklet2[1])
                        position = np.array([tracklet2[7], tracklet2[8], tracklet2[9]], dtype=np.float32)
                        targetDict[name] = Target(name, position=position, positionCS='OCS')  # wanted position: [x, y, z]

                # get person
                personName = 'Person {}'.format(tracklet[1])
                position = np.array(tracklet[7:10], dtype=np.float32)
                headpose = np.array([tracklet[12], tracklet[11], tracklet[10]], dtype=np.float32)
                headpose = np.concatenate([position, headpose])
                if model in ['gaussianModel', 'geometricalModel', 'gazeProbability', 'kalmanFilter']:
                    gaze = headpose.copy()
                    gaze[3:6] /= .8  # Dynamical Head Reference model
                    personDict[personName] = Person(personName, headpose=headpose, gaze=gaze, positionCS='OCS', poseCS='FCS', poseUnit='deg')  # wanted headpose: [x, y, z, yaw, pitch, roll]
                else:
                    personDict[personName] = Person(personName, headpose=headpose, positionCS='OCS', poseCS='FCS', poseUnit='deg')  # wanted headpose: [x, y, z, yaw, pitch, roll]

                # # Process (NOTE: both frameData-loops can be assemble using the following code, vfoaModule being
                # # called each time for one different person. It works as well)
                # identity = int(tracklet[1])
                # vfoaModule.compute_vfoa(personDict, targetDict, float(timestamp_current)/1000000000)
                # vfoa = vfoaModule.get_vfoa(personName)
                # estimation = vfoaModule.get_vfoa_best(personName)

            # Process
            vfoaModule.compute_vfoa(personDict, targetDict, float(timestamp_current)/1000000000)
            vfoa = vfoaModule.get_vfoa(normalized=False)
            estimation = vfoaModule.get_vfoa_best()

            for tracklet in frameData:
                identity = int(tracklet[1])
                personName = 'Person {}'.format(tracklet[1])
                # Compare with baseline (input csv file)
                vfoa_baseline = np.reshape(tracklet[13:], (-1, 2))
                vfoa_baseline = dict(zip(vfoa_baseline[:, 0], map(float, vfoa_baseline[:, 1])))
                vfoa_baseline_best, vfoa_baseline_best_prob = None, 0
                for key, val in vfoa_baseline.items():
                    if val > vfoa_baseline_best_prob:
                        vfoa_baseline_best, vfoa_baseline_best_prob = key, val

                nb_total[identity - 1] += 1
                if LABEL2CODE[estimation[personName].lower()] == LABEL2CODE[vfoa_baseline_best.lower()]:
                    nb_correct[identity - 1] += 1
                estimationList.append(LABEL2CODE[estimation[personName].lower()])
                baselineList.append(LABEL2CODE[vfoa_baseline_best.lower()])

                # Compare with ground truth
                if groundTruth is not None:
                    nb_total_gt[identity - 1] += 1
                    if LABEL2CODE[estimation[personName].lower()] == int(groundTruth['Person {}'.format(identity)][frameIndex]):
                        nb_correct_gt[identity - 1] += 1
                    groundTruthList.append(int(groundTruth['Person {}'.format(identity)][frameIndex]))

                # Prepare next step
                frameData = [dataline]
                timestamp_current = int(dataline[0])
        else:
            frameData.append(dataline)

    # Print results
    print('\nComparison with baseline (csvFile)')
    print('Total: {}'.format(nb_total))
    print('Correct: {}'.format(nb_correct))
    print('Accuracy: {}'.format(np.array(nb_correct) / np.array(nb_total)))
    print('Global acc: {}'.format(np.sum(nb_correct) / np.sum(nb_total)))

    if FRAME_BASED_EVALUATOR_AVAILABLE:
        evaluator = FrameBasedEvaluator()
        evaluator.evaluate(estimationList, baselineList)
        confMat = evaluator.get_confusionMatrix()
        print('Confusion Matrix:\n{}'.format(confMat))

    if groundTruth is not None:
        print('\nComparison with groundTruth (annotations)')
        print('Total: {}'.format(nb_total_gt))
        print('Correct: {}'.format(nb_correct_gt))
        print('Accuracy: {}'.format(np.array(nb_correct_gt) / np.array(nb_total_gt)))
        print('Global acc: {}'.format(np.sum(nb_correct_gt) / np.sum(nb_total_gt)))

        if FRAME_BASED_EVALUATOR_AVAILABLE:
            evaluator = FrameBasedEvaluator()
            evaluator.evaluate(estimationList, groundTruthList)
            metrics = evaluator.get_microMeasures()
            confMat = evaluator.get_confusionMatrix()
            print('Confusion Matrix:\n{}'.format(confMat))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read csv file, run vfoa Module on it and write csv result file')
    parser.add_argument('--inputDataFile', '-i', type=str, default='test_runVfoaModule_data.csv',
                        help='csv file, with lines: timestamp,id,x,y,w,h,conf,x,y,z,roll,tilt,pan,target,proba,target,proba,...')
    parser.add_argument('--annotations', '-gt', type=str, default='test_runVfoaModule_gt.npz', help='ground truth annotations (npz)')
    parser.add_argument('--method', '-m', type=str, default=None, help='method used in ["geometricalModel", "gaussianModel", "gazeProbability", "HMM"]')
    args = parser.parse_args()

    main(args.inputDataFile, annotations=args.annotations, model=args.method)
