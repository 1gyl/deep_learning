# ��������
�������ǻ������ṹ�����о��ߵġ�һ��ģ�һ�þ���������һ������㡢���ɸ��ڲ��������ɸ�Ҷ��㣻Ҷ����Ӧ�ھ��߽��������ÿ�����������������ϸ������Բ��ԵĽ�������ֵ��ӽ���У�������������ȫ�����Ӹ���㵽ÿ��Ҷ����·����Ӧ��һ���ж��������С���������Ŀ����Ϊ�˲���һ�÷�������ǿ��������δ��ʾ������ǿ�ľ������������������ѭ�򵥶�ֱ�۵ġ��ֶ���֮������
��������������һ���ݹ���̡�

���룺ѵ����D={(x1,y1),(x2,y2),������(xm,ym)};
    ���Լ�A={a1,a2,����,ad}
����:����TreeGenerate(D,A)
1. ���ɽڵ�node;
2.  if D������ȫ����ͬһ���C then
     ��node���ΪC���㣻 return 
    end if
3.  if A=$\emptyset$ OR D��������A��ȡֵ��ͬ then 
      ��node���ΪҶ��㣬�������ΪD�������������ࣻreturn
    end if
4.  ��A��ѡ�����Ż�������a*
    for a* ��ÿ��ֵ$a*^v$do
     Ϊnode����һ����֧����$D_v$��ʾD����a*��ȡֵΪ$a*^v$�������Ӽ�
     if $D_v$ Ϊ��then
      ����֧�����ΪҶ��㣬�������ΪD�����������ࣻreturn
     else
      ��TreeGenerate($D_v,A\{a*}$)Ϊ��֧���
    end if
    end for

�ھ����������㷨�У����������λᵼ�µݹ鷵�أ�
��1����ǰ������������ȫ����ͬһ������軮��
��2����ǰ���Լ�Ϊ�գ�������������������������ȡֵ��ͬ���޷�����
��3����ǰ����������������Ϊ�գ����ܻ���
��(2)�����εĵ�ǰ���ΪҶ��㣬����������趨Ϊ�ý����������������𣬵�(3)�����εĵ�ǰ���ҲΪҶ��㣬����������趨Ϊ�丸��㵫���������������ע�����������εĴ���ʵ�ʲ�ͬ������(2)�������õ�ǰ���ĺ���ֲ�������(3)���ǰѸ����������ֲ���Ϊ��ǰ��������ֲ�

# ����ѡ��
�������Ĺؼ���ѡ�����Ż������ԡ�һ����ԣ����Ż��ֹ��̵Ĳ��Ͻ��У�����ϣ���������ķ�֧�������������������������ͬһ��𣬼����ġ����ȡ�(purity)Խ��Խ��

## ��Ϣ����
����Ϣ�ء�(information entropy)�Ƕ����������ϴ�����õ�һ��ָ�ꡣ�ٶ���ǰ��������D�е�k��������ռ�ı���Ϊpk(k=1,2,����,|y|)����D����Ϣ�ض���Ϊ

$Ent(D)=-\sum_{k=1}^{|y|}p_klog_2p_k$

Ent(D)��ֵԽС����D�Ĵ���Խ��
�ٶ���ɢ����a��V�����ܵ�ȡֵ${a^1,a^2,a^3,����,a^V}$,��ʹ��a����������D���л��֣�������V����֧��㣬���е�v����֧��������D������������a��ȡֵΪ$a^v$����������Ϊ$D^v$�����ǵ���ͬ�ķ�֧�������������������ͬ������֧��㸳��Ȩ��$|D^v|/|D|$����������Խ��ķ�֧����Ӱ��Խ�󣬿ɼ��㴦������a��������D���л�������õ�"��Ϣ����"(information gain)

$Gain(D,a)=Ent(D)-\sum_{v=1}^V \frac{|D^v|}{D}Ent(D^v)$

һ����ԣ���Ϣ����Խ����ζ��ʹ������a�����л������õġ�����������Խ��
���ǿ�������Ϣ���������о�������������ѡ��
$a*=arg maxGain(D,a)$��

## ������
��Ϣ����׼��Կ�ȡֵ��Ŀ�϶����������ƫ�ã�Ϊ��������ƫ�ÿ��ܴ�������Ӱ�졣������C4.5�������㷨��ֱ��ʹ����Ϣ���棬����ʹ�á������ʡ�(gain ratio)��ѡ�����Ż�����
�ԡ��䶨��Ϊ

$Gain_ratio(D,a)=\frac{Gain(D,a)}{IV(a)}$

$IV(a)=-\sum_{v=1}^{V}\frac{|D^v|}{|D|} log2\frac{|D^v|}{|D|}$

��Ϊ����a��"�̶�ֵ"(intrinsic value)������a�Ŀ���ȡֵ��ĿԽ��(��V)Խ����IV(a)��ֵͨ����Խ��
��Ҫע����ǣ�������׼��Կ�ȡֵ��Ŀ���ٵ���������ƫ�ã���ˣ�C4.5�㷨������ֱ��ѡ����һ�����ĺ�ѡ�������ԣ������ȴӺ�ѡ�����������ҳ���Ϣ�������ƽ��ˮƽ�����ԣ��ٴ���ѡ����������ߵ�

## ����ָ��
CART������ʹ��"����ָ��"(Gini index)��ѡ�񻮷����ԣ����ݼ�D�Ĵ��ȿ��û���ֵ������

$Gini(D)=\sum_{k=1}^{|y|}\sum_{k'!=k}p_kp_{k'}=1-\sum_{k=1}^{|y|}p_k^2$

Gini(D)��ӳ�˴����ݼ�D�������ȡ����������������ǲ�һ�µĸ��ʡ���ˣ�Gini(D)ԽС�������ݼ�D�Ĵ���Խ��
����a�Ļ���ָ������Ϊ

$Gini_index(D,a)=\sum_{v=1}^V\frac{|D^v|}{D}Gini(D^v)$

���ǣ������ں�ѡ���Լ���A�У�ѡ���ĸ�ʹ�û��ֺ����ָ����С��������Ϊ���Ż������ԣ���a*=argmin Gini_index(D,a)

## ��֦����
��֦(pruning)�Ǿ�����ѧϰ�㷨�Ը�(������ϡ�)����Ҫ�ֶΡ��ھ�����ѧϰ�У�Ϊ�˾�������ȷ����ѵ����������㻮�ֹ��̽������ظ�����ʱ����ɾ�������֧���࣬�����ڰ�ѵ����������һЩ�ص㵱���������ݶ����е�һ�����ʶ����¹���͡���ˣ���ͨ������ȥ��һЩ��֧�뿪���͹���ϵķ���
��������֦�Ļ��������С�Ԥ��֦(prepruning)���͡����֦(post-pruning)����Ԥ��֦��ָ�ھ��������ɹ����У���ÿ������ڻ���ǰ�Ƚ��й��ƣ�����ǰ���Ļ��ֲ��ܴ�������������������������ֹͣ���ֲ�����ǰ�����ΪҶ��㣻���֦�����ȴ�ѵ��������һ�������ľ�������Ȼ���Ե����ϵضԷ�Ҷ�����п��죬�����ý���Ӧ�������滻ΪҶ����ܴ������������������������򽫸������滻ΪҶ��㡣
���������������жϾ��������������Ƿ�����

### ǰ��֦
�ڻ���֮ǰ���������������ڸ���㡣�������л��֣��ý�㽫�����ΪҶ��㣬�������Ϊѵ���������������Ԥ��֦ʹ�þ������ĺܶ��֧��û�С�չ�������ⲻ�������˹���ϵķ��գ������������˾�������ѵ��ʱ�俪���Ͳ���ʱ�俪������һ���棬��Щ��֧�ĵ�ǰ������Ȼ���������������ܡ��������ܵ��·���������ʱ�½�������������Ͻ��еĺ�������ȴ���ܵ��·���������ʱ�½�������������Ͻ��к�������ȴ�п��ܵ��·�������ʱ�½�������������Ͻ��еĺ�������ȴ���ܵ���������������ߣ�Ԥ��֦���ڡ�̰�ġ����ʽ�ֹ��Щ��֧չ������Ԥ��֦������������Ƿ��ϵķ���

### ���֦
���֦�ȴ�ѵ��������һ�����������������֦������ͨ����Ԥ��֦�����������˸���ķ�֧��һ�������£����֦��������Ƿ��Ϸ��պ�С������������������Ԥ��֦�������������֦��������������ȫ������֮����еģ�����Ҫ�Ե����ϵض����е����з�Ҷ��������һ���죬�����ѵ��ʱ�俪����δ��֦��������Ԥ��֦��������Ҫ��ö�

## ������ȱʧֵ

### ����ֵ����
�������ԵĿ�ȡֵ��Ŀ�������ޣ���ˣ�����ֱ�Ӹ����������ԵĿ�ȡֵ���Խ����л��֡���ʱ������������ɢ�������������ó�����򵥵Ĳ����ǲ��ö��ַ�(bi-partition)���������Խ��д���������ʽC4.5�������㷨�в��õĻ���
����������D����������a,�ٶ�a��D�ϳ�����n����ͬ��ȡֵ������Щֵ��С����������򣬼�Ϊ${a^1,a^2,����,a^n}$�����ڻ��ֵ�t�ɽ�D��Ϊ�Ӽ�$D_t^-$��$D_t^+$������$D_t^-$������Щ����aȡֵ������t����������$D_t^+$�������Щ����aȡֵ�ϴ���t�������ء���Ȼ�������ڵ�����ȡֵ$a^i��a^{i+1}$��˵��t������$[a^i,a^{i+1})$��ȡ����ֵ�������Ļ��ֽ����ͬ����ˣ�����������a�����ǿɿ������n-1��Ԫ�صĺ�ѡ���ֵ㼯��

$T_a={\frac{a^i+a^{i+1}}{2}|1<=i<=n-1}$

��������$[a^i,a^{i+1})$����λ��$\frac{a^i+a^{i+1}}{2}$��Ϊ��ѡ���ֵ㡣Ȼ�����ǿ�������ɢ����ֵһ����������Щ���ֵ㣬ѡ�����ŵĻ��ֵ�����������ϵĻ���

$Gain(D,a)=max Gain(D,a,t)=max Ent(D)-\sum_{\lambda \in{-,+}} \frac{|D_t^\lambda|}{|D|}Ent(D_t^\lambda)$

����Gain(D,a,t)��������D���ڻ��ֵ�t���ֺ����Ϣ���档���ǣ����ǾͿ�ѡ��ʹGain(D,a,t)��󻯵Ļ��ֵ�

��ע����ǣ�����ɢ���Բ�ͬ������ǰ��㻮������Ϊ�������ԣ������Ի�����Ϊ�������Ļ�������
