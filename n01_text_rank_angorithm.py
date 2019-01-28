from collections import defaultdict
import numpy as np
import jieba

text = """
近期，各个企事业单位开始陆续发放新年第一个月的工资薪金了。按照新修订的《个人所得税法》，到手的工资收入和缴纳的个税金额，和改革之前相比有什么变化呢？
月收入万元 大都无需再缴个税
在重庆一家乳品生产企业，公司财务部员工唐君，刚刚领到1月份的工资条。
重庆市某乳业公司财务部员工 唐君：实行新的个税法以前，我2018年8月份税前金额大概接近7000元，要缴30几元到40几元钱的个税，到10月份实行5000元个税免征额之后，我就不会存在缴纳个人所得税的情况了。
个税改革前：发放工资7000元，减除单位预扣的三险一金（专项扣除）、企业年金等（其他扣除）共2000元左右，再减除3500元的个税起征点（个税免征额），余下1500元应纳税所得额，对应税率是3%，需要缴纳个税45元。
个税改革后：由于个税起征点提高到5000元，余下的应纳税所得额为0，唐君就不需要缴纳个税了。
对于月收入近1万元的车间管理人员，由于新修订的《个税法》增加了专项附加扣除，他还能进一步享受到赡养老人、子女教育共3000元的税前扣除。
重庆市某乳业公司生产车间 王利：2018年9月份的时候，个税大概缴300多元。到2018年10月份左右，就是100多元。到2019年，我可以享受专项附加扣除3000元左右，就没有个税了。到手的实收金额，每个月大概要比以前多200元左右。
个税改革前：发放工资10000元，减除单位预扣的三险一金（专项扣除）、企业年金等（其他扣除）共2000元左右，再减除3500元的个税起征点（个税免征额），余下4500元应纳税所得额，对应的税率是3%和10%，需要缴纳个税345元。
个税改革后：不仅起征点提高到5000元，王利还可以减除3000元左右的专项附加扣除，余下的应纳税所得额为0，也不需要缴纳个税了。
从全国来看，2018年10月，个税起征点由3500元提高至5000元，就减少了6000多万名纳税人；2019年，增加六项专项附加扣除，预计又将减少数千万名纳税人。
国家税务总局所得税司个税一处处长 阳民：这次个税改革，它对中低收入者减税减负的效果是最明显的。因为新税法提高了起征点，有相当一部分纳税人能够享受到每个月2000元或3000元的专项附加扣除。这样算起来再减去三险一金，那么月收入在10000元钱左右的原纳税人，可能就不用缴纳个税了。
年收入40万元以内 减税力度最大
与劳动密集型企业不同，收入状况更好一些的金融企业，在个税改革后，到手的工资收入和缴纳的个税金额，又有什么变化呢？
在重庆的一家金融企业，记者随机采访了一位普通员工。个税改革后几个月，陈女士的纳税金额发生了较大的变化。
重庆某融资担保集团员工 陈美灵：2018年9月，我缴纳个税800多元，但是在10月份，我缴纳个税就只缴纳了300多元，再到了今年1月份，就没有再缴纳个人所得税。
小陈算了一笔账，按照新个税法，她一个月可以省税800多元，由于每个月工资差不多，一年就能省税近万元。
公司的管理层人员，个税的下降幅度就更为明显。
重庆某融资担保集团贷款担保部总经理助理 蒋睿凌：去年9月，我缴纳的个税在2600元左右，到10月份基数调整之后，缴纳个税在1800元左右，今年1月份，我增加了3000元钱的专项附加扣除。
专项附加扣除具有叠加效应，不仅总体上提高了免税额度，还拉低了他的应税税率。由于月度工资差不多，现在，他月均收入的应税税率从最高25%下降到了20%。
由于新个税法扩大了3%、10%、20%三档税率的级次，相当于拉低了这些级次收入的税率。相比改革前，工薪收入扣除免税额度后，月应纳税所得额在1500元-25000元范围的纳税人，获益最大。
国家税务总局所得税司个税一处处长 阳民：这三档税率级距，它对应的主要是年收入40万以下的纳税人群体，这一纳税人群体，相对应来说的话，它享受的减税力度是最大的。
从全国来看，2018年个税改革施行3个月，减税就达1000亿元；2019年增加专项附加扣除后，个税减税规模预计全年超过3000亿元。
专项附加扣除 体现个税调控差异性
在采访中，一些纳税人已经开始感受到，专项附加扣除是此次个税改革最大的亮点，它体现了个税扣除量能负担的差异化原则。
业内专家表示，起征点即个税基本免征额，是基于社会生活成本的基本扣除，是普适性的。专项附加扣除虽然相当于再次提高了起征点，但是它的调控目标是不一样的。
国家税务总局所得税司个税一处处长 阳民：这一次的税制改革当中，把这些支出考虑进去其实也是考虑到了一个普适性和特殊性相结合的税制考虑因素，从而能够更加进一步增强税制的公平和合理性。"""


# 获取
def get_word_confidence():
    """获取词之间的共现度字典"""
    global co_dict
    global word_all
    stopwords = {line.strip(): 1 for line in open('./data/stopwords.txt', 'r', encoding='utf-8').readlines()}

    sentence_li = [i.lstrip().rstrip() for i in text.split('。')]

    co_tuple_dict = defaultdict(int)
    num_dict = defaultdict(int)
    for sentence in sentence_li:
        word_li = [i for i in jieba.cut(sentence) if not stopwords.get(i, None)]
        for index in range(100):
            new_word_li = word_li[index:index + 5]
            if len(new_word_li) == 5:
                for a in new_word_li:
                    num_dict[a] += 1
                    for b in new_word_li:
                        if a != b:
                            co_tuple_dict[(a, b)] += 1

    print(co_tuple_dict)
    print(num_dict)
    co_dict = dict()
    for tuple_ab, num in co_tuple_dict.items():
        co_dict[tuple_ab] = num / num_dict[tuple_ab[0]]

    word_all = num_dict.keys()
    print(word_all)
    print(len(word_all))
    return co_dict, word_all


def get_square_matrix():
    """根据共现度字典，获取textrank方针"""
    global li_np
    li = []
    for word in word_all:
        li2 = []
        for word2 in word_all:
            cow = co_dict.get((word, word2), 0)
            li2.append(cow / 4)
        li.append(li2)
        # print(sum(li2))
    print(li)
    li_np = np.array(li)
    return li_np


def calculate_converge_list():
    """初始化列表，并与方针相乘，直到收敛"""
    global M, U
    M = li_np.T
    U = [1 / len(word_all) for i in word_all]
    U0 = np.array(U)
    print(U0)
    U_past = []
    while True:
        # U = np.dot(M, U)
        U = 0.85 * (np.dot(M, U)) + 0.15 * U0
        # print('Un: ', U)
        if str(U) == str(U_past):
            break
        U_past = U
        # print(U)

    print('U converge to: ', U)
    print(list(zip(word_all, U)))
    li = sorted(list(zip(word_all, U)), key=lambda x: x[1], reverse=True)
    print(li)
    return li


def get_combine_word():
    """获取排序后文章中前n个词的两两在一起的组合"""
    for w1 in sorted_li[:15]:
        for w2 in sorted_li[:15]:
            if w1[0] + w2[0] in text:
                print(w1[0] + w2[0])
            for w3 in sorted_li[:15]:
                if w1[0] + w2[0] + w3[0] in text:
                    print(w1[0] + w2[0] + w3[0])


if __name__ == '__main__':
    co_dict, word_all = get_word_confidence()
    li_np = get_square_matrix()
    sorted_li = calculate_converge_list()
    get_combine_word()
