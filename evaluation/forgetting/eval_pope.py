import argparse
import datetime
import json
import os

def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')

        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    return f1, acc, precision, recall, yes_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--save_to_txt", type=str)
    args = parser.parse_args()

    print(args.annotation_dir)
    print(args.question_file)
    print(args.result_file)

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]

    with open(args.save_to_txt, 'a') as f:
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write(args.result_file + '\n')
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Evaluation on POPE on time: {current_time}\n")
        for file in os.listdir(args.annotation_dir):
            if file.startswith('coco_pope_'):
                if file.endswith('.json'):
                    category = file[10:-5]
                    cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
                    f1, acc, precision, recall, yes_ratio = eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
                    f.write("====================================\n")
                    f.write('Category: {}, # samples: {}\n'.format(category, len(cur_answers)))
                    f.write('TP\tFP\tTN\tFN\n')
                    f.write('%.3f, %.3f, %.3f, %.3f, %.3f\n' % (f1, acc, precision, recall, yes_ratio))
        f.write("====================================\n")
    print('Results saved at:', args.save_to_txt)