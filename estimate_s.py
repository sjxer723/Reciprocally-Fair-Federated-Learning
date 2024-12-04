import argparse
from MNIST.shapley_value import FLInstance
import numpy as np
from random import randint


def a_cifar(sum_of_s, i):
    if i < 50:
        _alpha, _beta = 1.407, 0.281
    else:
        _alpha, _beta = 1.256, 0.265
    return 1 - (_alpha * 1.0) / pow(sum_of_s, _beta)

def a_mnist(sum_of_s):
    _alpha = 0.384
    _beta = 0.438
    return 1 - (_alpha * 1.0) / pow(sum_of_s, _beta)


def evaluate_rec_mnist(method: str):
    cost_per_s = 0.00045
    cost_scalar_beta = 0.8
    
    alpha_1 = 0.384
    beta_1 = 0.438

    s_all_br_bg = 30
    s_all_br = 9

    s_br_dict = {6: 1.5093212248636916, 9: 1.032781342820187, 0: 0.0006836612343466197, 2: 2.238653242474003, 4: 0.33532293444981964, 3: 2.677144224655713, 5: 1.787404804018398, 1: 1.572782906378707, 8: 1.11920583095249, 7: 0.14051884737959533}
    s_br_bg_dict = {i: 3 for i in range(10)}
    
    fl = FLInstance(10, [1 for _ in range(10)], alpha_1, beta_1)
    if method == "br":
        fl.update_s(s_br_dict)
    elif method == "br-bg":
        fl.update_s(s_br_bg_dict)
    
    shapley_vals = []
    rec = []
    for i in range(10):
        shapley_val = fl.compute_shapley_value(i)
        shapley_vals.append(shapley_val)
        if method == "br":
            rec.append(a_mnist(s_all_br) / shapley_val)
        elif method == "br-bg":
            rec.append((a_mnist(s_all_br_bg) + cost_per_s * cost_scalar_beta - \
                        cost_per_s * cost_scalar_beta * (sum([int(x) for x in s_br_bg_dict.values()]) - int(s_br_bg_dict[i]))/99.0) / shapley_val)
        
    print(shapley_vals)
    min_rec = 10
    for r in rec:
        if r < 0:
            continue
        min_rec = min(min_rec, r)
    print(min_rec)

def evaluate_rec_cifar10(method: str):
    cost_per_s = 0.00045
    cost_scalar_beta = 0.8
    
    alpha_1 = 1.407
    beta_1 = 0.281
    alpha_2 = 1.256
    beta_2 = 0.246

    s_all_br_bg = 199
    s_all_br = 103

    s_br_dict = {0: 1.683469552951641, 1: 1.8421765591412627, 2: 1.7900731233319431, 3: 1.5290170268501064, 4: 1.8605020406730604, 5: 1.862100754489337, 6: 1.8513186764819316, 7: 1.6909884760772322, 8: 1.560849787765835, 9: 1.6255175441814038, 10: 1.3968085423526437, 11: 1.8746482449823991, 12: 1.8759779547842348, 13: 1.7908009453483604, 14: 1.5621290247054256, 15: 1.5176385599607338, 16: 1.3841927646933074, 17: 1.441319547503387, 18: 1.5620758861713866, 19: 1.8044076297365108, 20: 1.7293853839699647, 21: 1.8821191442046263, 22: 1.5982897074061515, 23: 1.6593030405671538, 24: 1.7480547538645466, 25: 2.1570320510287533, 26: 1.7768896796386418, 27: 1.646728014994374, 28: 1.5618833423765295, 29: 1.616665775763417, 30: 1.3584563496954365, 31: 1.7246251840538473, 32: 1.625418577495549, 33: 1.7153463031965526, 34: 1.723772153525631, 35: 1.740363233386682, 36: 1.768915899402433, 37: 1.8901285866032422, 38: 2.0604380058331264, 39: 1.6458893009527549, 40: 1.9281780821348113, 41: 1.5745939975121743, 42: 1.6188217664187399, 43: 1.8562758238478674, 44: 1.5555689128841044, 45: 1.5060884834458115, 46: 1.955405426752168, 47: 1.3506433834488096, 48: 1.4621368135696027, 49: 1.702473181832124, 50: 1.974210661585155, 51: 1.4637153824041145, 52: 1.862277984584242, 53: 1.9416143965881008, 54: 1.8419490411845465, 55: 1.6688480930477614, 56: 1.5581951991103986, 57: 1.608444208047019, 58: 1.615972146539191, 59: 1.610763319897366, 60: 1.5810586845164902, 61: 1.702142446110523, 62: 1.4648331324145076, 63: 1.7987597754982056, 64: 1.4897441776853924, 65: 1.6058182998155763, 66: 1.5533997664723025, 67: 1.7310637888285838, 68: 1.681571797664757, 69: 1.7054256325461477, 70: 1.5987954237792252, 71: 1.5802307058308465, 72: 1.5795552454039734, 73: 1.7136282824202946, 74: 1.4667512165764842, 75: 1.6659003767730418, 76: 1.468389581522243, 77: 2.0970468379420577, 78: 1.8565726571795436, 79: 1.5389548697162339, 80: 1.6440317813236764, 81: 1.6778942083393955, 82: 1.4913469091849665, 83: 1.9640918195136043, 84: 1.7896302979416137, 85: 1.780836469442848, 86: 1.497213227971893, 87: 1.9402108353030898, 88: 1.622248769768983, 89: 1.485252198304548, 90: 1.5995914862217169, 91: 1.5343092053901763, 92: 1.7885047164828127, 93: 1.8421544400433516, 94: 1.8725773663361343, 95: 1.5720152358799298, 96: 1.652247007276204, 97: 1.95873694391527, 98: 1.774216076090838, 99: 1.443639520216086}
    s_br_bg_dict = {0: 2.1708863796810602, 1: 2.3873049105158795, 2: 2.816272290768819, 3: 2.6954484115071202, 4: 2.6956576418618274, 5: 2.4018671882224822, 6: 2.1865171089260755, 7: 2.2208526158320945, 8: 2.726121377596669, 9: 2.8069181901418387, 10: 1.714262511532768, 11: 2.3146221090353754, 12: 2.8595563364749106, 13: 2.389631193285626, 14: 2.3520380190931496, 15: 2.4217076183310415, 16: 2.420053855901638, 17: 2.2091048738181414, 18: 2.4036705498424698, 19: 1.8324125084387954, 20: 2.643655418034063, 21: 2.276100606334784, 22: 2.491066379527589, 23: 2.3890803996523884, 24: 2.20674373065713, 25: 2.395174279316161, 26: 2.9504965292167906, 27: 2.7131810147425344, 28: 2.16468575740499, 29: 2.5888132134152153, 30: 2.182237391843615, 31: 2.3302803226789086, 32: 2.7010662975438255, 33: 2.763293879182683, 34: 2.3817159132327688, 35: 2.296224113166021, 36: 2.5073028438208613, 37: 2.382495666579506, 38: 1.9644369270768332, 39: 2.3004224737419854, 40: 2.1676196148469775, 41: 2.4948428017581055, 42: 2.2970979530706326, 43: 2.545418149411403, 44: 2.399727492872052, 45: 2.6152140851057064, 46: 2.8674228930499672, 47: 2.491070552768109, 48: 2.104009294340959, 49: 2.7971342493173714, 50: 2.6720695224116273, 51: 2.676018068643278, 52: 2.791893417149307, 53: 2.63035602669403, 54: 2.2642701000618475, 55: 2.2480497074817842, 56: 2.423562015981916, 57: 2.509276209139834, 58: 2.528053165437967, 59: 2.224763445152612, 60: 2.3988520752374396, 61: 3.021199003416216, 62: 2.0061253339954987, 63: 2.3457458072554496, 64: 2.715883759269915, 65: 2.371970359336969, 66: 2.292063189209198, 67: 2.449583942763328, 68: 2.2777619842356516, 69: 2.2160226003889254, 70: 2.48263045941322, 71: 2.2743575001839034, 72: 2.1872723856200107, 73: 2.353831507586725, 74: 2.441828612034389, 75: 2.40120741332551, 76: 2.3981714049961944, 77: 2.454498797793282, 78: 2.4143975216101343, 79: 2.677722497678261, 80: 2.0228579456823548, 81: 2.1369604039113375, 82: 2.360732192549474, 83: 2.7967309753874847, 84: 2.450659655873786, 85: 2.3922080738026734, 86: 2.649193654582787, 87: 2.3176292521502257, 88: 2.5398231386606684, 89: 2.7188955863978967, 90: 2.2329389561408837, 91: 2.445987036006409, 92: 2.420584176451213, 93: 2.1590437052742084, 94: 2.3554049754268203, 95: 2.1077568817191525, 96: 2.047489688963259, 97: 2.240075544524646, 98: 2.2588563033028053, 99: 3.025256951184844}

    fl1 = FLInstance(100, [1 for _ in range(10)], alpha_1, beta_1)
    fl2 = FLInstance(100, [1 for _ in range(10)], alpha_2, beta_2)
    if method == "br":
        fl1.update_s(s_br_dict)
        fl2.update_s(s_br_dict)
    elif method == "br-bg":
        fl1.update_s(s_br_bg_dict)
        fl2.update_s(s_br_bg_dict)
    
    shapley_vals = []
    rec = []
    for i in range(100):
        shapley_val = (fl1.compute_shapley_value(i) + fl2.compute_shapley_value(i))/2
        shapley_vals.append(shapley_val)
        if method == "br":
            rec.append(a_cifar(s_all_br, i) / shapley_val)
        elif method == "br-bg":
            rec.append((a_cifar(s_all_br_bg, i) + cost_per_s * cost_scalar_beta - \
                        cost_per_s * cost_scalar_beta * (sum([int(x) for x in s_br_bg_dict.values()]) - int(s_br_bg_dict[i]))/99.0) / shapley_val)
        
    print(shapley_vals)
    print(min(rec))


def evaluate_rec_fashionMnist(method: str):
    cost_per_s = 0.00045
    cost_scalar = 0.99
    
    alpha = 1.319
    beta = 0.219

    s_br_dict = evaluate_s_sum("br", alpha=alpha, beta=beta, cost_scalar=cost_scalar)
    s_br_bg_dict = evaluate_s_sum("br-bg", alpha=alpha, beta=beta, cost_scalar=cost_scalar)
    
    s_all_br_bg = sum(s_br_dict.values())
    s_all_br = sum(s_br_bg_dict.values())

    fl = FLInstance(100, [1 for _ in range(10)], alpha, beta)
    if method == "br":
        fl.update_s(s_br_dict)
    elif method == "br-bg":
        fl.update_s(s_br_bg_dict)
        
    shapley_vals = []
    rec = []
    for i in range(100):
        print(i)
        shapley_val = fl.compute_shapley_value(i)
        shapley_vals.append(shapley_val)
        if method == "br":
            rec.append(a_cifar(s_all_br, i) / shapley_val)
        elif method == "br-bg":
            rec.append((a_cifar(s_all_br_bg, i) + cost_per_s * cost_scalar - \
                        cost_per_s * cost_scalar * (sum([int(x) for x in s_br_bg_dict.values()]) - int(s_br_bg_dict[i]))/99.0) / shapley_val)
        
    print(shapley_vals)
    print(min(rec))

## Predict the value of ||s_i|| after FL terminates
def evaluate_s_sum(method:str, alpha=1.433, beta=0.265, cost_scalar=0.8):
    cost_per_s = 0.0005
    delta = 100

    def a_derivative(s_all):
        return (alpha * beta * 1.0) / pow(sum(s_all), beta + 1)
    
    s_dict = {i: 1 for i in range(100)}
    fl = FLInstance(100, [], alpha, beta)
    fl.update_s(s_dict)
    for _ in range(400):
        for _ in range(10):
            k = randint(0, 99)
            if method == "br":
                s_ = s_dict[k] + (a_derivative(s_dict.values()) -  cost_per_s) * delta
            elif method == "br-bg":
                s_ = s_dict[k] + (a_derivative(s_dict.values()) - (1 - cost_scalar ) * cost_per_s) * delta
            elif method == "br-sv":
                s_ = s_dict[k] + (fl.compute_shapley_value_derivative(k) - cost_per_s)*delta
            else:
                raise ValueError("Unknown method")
            s_dict[k] = s_
    print(s_dict)
    print(sum([int(x) for x in s_dict.values()]))
    
    return s_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", dest="method", default="br")
    parser.add_argument("--statics", dest="statics", default="s-sum")
    parser.add_argument("--dataset", dest="dataset", default="cifar")
    parser.add_argument("--alpha", dest="alpha", default=1.433)
    parser.add_argument("--beta", dest="beta", default=0.265)

    args = parser.parse_args()

    if args.statics == "s-sum":
        evaluate_s_sum(args.method, float(args.alpha), float(args.beta))
    elif args.statics == "rec":
        if args.dataset == "cifar":
            evaluate_rec_cifar10(args.method)
        elif args.dataset == "minist":
            evaluate_rec_mnist(args.method)
        else:
            evaluate_rec_fashionMnist(args.method)

if __name__ == "__main__":
    main()