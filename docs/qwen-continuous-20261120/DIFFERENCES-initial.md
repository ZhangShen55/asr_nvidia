# 初版128段文本差异

静态HF与连续生成的同源配对，在线复读保护关闭。128段中124段原始文本相同、4段不同。以下为完整原始输出，未指定任一路径为真值；static结果未保存token，因此不能宣称token相同或不相同。

## index 63 / vad063

来源：/home/zhangs/asr_nvidia/test/test_audio/金匮中医/金匮中医（中文）.wav，290.52–298.53秒。

静态：

language Chinese<asr_text>就会挟寒气，所以带来什么结果呢？第十篇腹满寒疝数时，叫做寒饮逆满之症，附子粳米汤症。

连续：

language Chinese<asr_text>就会挟寒气，所以带来什么结果呢？第十篇腹满寒疝宿食，叫做寒饮逆满之症，附子粳米汤症。

## index 73 / vad073

来源：/home/zhangs/asr_nvidia/test/test_audio/课程/反常积分与级数（中法）.wav，2310.41–2314.19秒。

静态：

language French<asr_text>Logarithme de n, c'est aussi au signe de croissance comparée, hein.

连续：

language French<asr_text>Logarithme de n, c'est aussi au signe de croissance comparée, mais

## index 113 / vad113

来源：/home/zhangs/asr_nvidia/test/test_audio/课程/反常积分与级数（中法）.wav，1812.56–1820.57秒。

静态：

language French<asr_text>Regardez, ça, ça veut dire quoi ? Ça veut dire que vn est négligeable devant un. D'accord ? Ça veut dire vous avez

连续：

language French<asr_text>Regardez ça, ça veut dire quoi ? Ça veut dire que vn est négligeable devant un. D'accord ? Ça veut dire vous avez

## index 123 / vad123

来源：/home/zhangs/asr_nvidia/test/test_audio/课程/多模态机器学习2(中英).wav，2823.44–2831.45秒。

静态：

language Chinese<asr_text>太神奇了！这个我来，然后我我们所谓的商业运作嘛，但是我们。

连续：

language Chinese<asr_text>太神奇了！就是我，就是我，然后我们所谓的“少油”一个，但是我们。
