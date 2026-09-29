
import argparse
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import zlib

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

CREDIT_SHA256 = 'fff49bc186cbddb3ace7371d40d9fbbb3af4f126019c13ff3f562249b1454f4d'
CREDIT_B85 = (
    'c-oD9+p-<Gjph6NN^kApoy@26Eh?#G$r`z>R!ynKUq2-7AakoG?1&Dzx$`hV5SN7o<Zq80o+CfP<NrMV@%Z`p^AVq^KK_{h3Gwl1'
    '{%;|qkmmpYzy8<zM;*`BALUsZEdT!a{K(HxaCxKUq2xzx_=8lQH9W$z%KG?RPyaL@NDm+3F;bbUe|x09wlY@~*-d_S+Rgeu>jyD}'
    'N5<vh8T+F?TdHRFJ<9WctmP@d=6`$S^bF&HkN!57CfNx-I5%0D*MsAFv$gnt=C<3mIJWZ0F^!6^nVwgA#JTf<Tlw%|{(*p^(RU5='
    'U1Kg6{E<4vm_RiD_wnz?R}kgZ-1<1z=ZLZ?X-Tp?uZTZD;l$>*73$nZna>6Jzzf6i!k*@zNbrkK_C?a5w@w$l#9UrweR<}*o!697'
    '3V6DCEL}le__~De_Yzp1Wvq=fmgiTN%_l3HxjMBAA6zOZg6hqm57L5uJUlnj$;gsiPbVFYa=4-f2{t{Y(yORKK*`VNXI?-52U6j@'
    ')>m2o9R=U{`E*qC`KeU=3lVPAlhpF#lw)|d{iNsx-aq2Co;IxRtRHlFV-gVUXg<F_=Z|CLKkz}6d2Xo7rJYOC6g&I^-XY5B0EcX^'
    'NtW%DzW(%(wH@A^Zn*9}Ccff*>2WM{R)3*Q7@G}Q@`(}uKwR(J%-IIoyhuMzNs@3X%2I?QCS)#8{BXPZwKr7wFOT2mHKRK+mtTkO'
    ')VYiCh~vp8HGdFAkjH9xQ(pZ0+)3wcf7&%3uAW$spQ^wt!@Bkfzo5If4yZi)1wVNIe*HpBtJuCa%pVAki`OrycTwVT?&n(8ql*|p'
    'H26<=*=k<42EN{HX~Gm>;y1>(@Jd{G`7}36n4r0OvL$Zz8~#KK-r$Rr67DC`;*zJCt-N-L$X}C4iadXkSg`YnJ!ZbUcvHG6XHe~-'
    'tG+f=Bma?JN!^w)Av>Nng<HB`>FA~%pRawvi>|-%>C^geEb`{^GV?DzGW{x-bqV0+^F=Gq?M=WWrpQ)n8E3r&l`Jzn-<Z%eGHX;@'
    'K8&~Z(-{?U?FF<Dw{*ezWpFWk9s|6qzx~}@pLrps0|8O3(;M^-#hUe4Sx`AY8lMHaBYwIHX}!cKY$?{i)-_mv;XAxSq+cX97Nt(L'
    'NV(xl#3d(*P(VqV+#|lIdHy**iTOw7)sUbAZbp7q=bTa$+nd+((VVL_Rj0Zeq-m>Bq2!$d*;Bnm<%<0D2Kp-3CqB2OD_Ww{0%Me{'
    'g)#94J$PyMP>s71_z79>ADu69?67u9IjaaeY4e?TpCXy(IxpEeOAbw}hhf{L+MmFI1{W#~A7>cb1$x6G{IYc4$88$n)%95Gi9q@s'
    '?kqpoBg?vEeaEF)ZM#gvXn9{6#_sl;3z7?qPXm$&H?M(Kpj(*{1G^wnsAY|_1Q-wqPE(Fr{-^bDf5>hM+u4BYIZmS=>>k$~5JgDI'
    '#HRCK7q9<AJhZh<awA|p{L8Jc30;Nw3ensT){g6DN^KmzKu(Zd0G6u1qRYg_8YzI8G17eJ_EcLI*A@2x(v(rUYB8#i)Oq~{ziFD='
    '&|6HkW=lR3gsEs<{QC+9NUo|t1;q3Y#WZbgF_DAMDBfTlM9I^x8rjFZB@@@+XO@D&snN8cHULAl63pu5?8N-y`#D3jqIFn1wWh6P'
    'DP6`n=iMK~X5J2B_Nws++g@W}N~V{@AXXhG{lH4kKR@9uflI<+I|s44opq$LzCyrb?QZfmBNmYi`m)P-<gi_jtI*b`&xf6;%##m|'
    'A?SLC9&%)NI&KR;z(5ar>>GFyd;g8ek|V^ZCP)rTa$7#l8E5&ywEN4(kq}mZ>(n!eD!rL2_7N>|_iL}jFE;PV6v4D`Q4g<dspv3@'
    '5yZ&CQlIxFI)(S^6haWm{&W7vyPV}FQ-`NENN?v2>J)exGfz8+iFXho<ET%o3ftnOVlWCho_9`WEvs<%BS?(@{)HySpTgSnI}FG='
    'grny6ogOmHV+9=p8<1U)5q3zSM?MY+TZ1^%<M!$8h;r6dRl+;pAL1k5QIrm;82cKpnW?e<t}VM*#CxQ;tVs}^gb+$(2YW~-Vp8}A'
    '6=L~jq<11ExHJOfRPT&dGN+gaD>CG!f-@u&yQ`n8eD}bBuPuDxV8OfH6}zE+fT<Oo`~C4BGAKz-V?i(3D+fmaUpE!CH}$>b^XBfo'
    'ROb>fiX+VcueM#P-qa>T?CDHqzji#Wh&;fCXA>djJu%PIebA;0o;k(lWMiM37Y7wBc*PSWO3_82M=YEgOC@Kw`>)_({ce(&1oUo^'
    '#e%7!C`vZ3nxY7O84-yOFpA=!C>az*bbLOpD~>5KTDN&58<Qy;_MPkS8$UT#n;69ftrE1bJUK`6;P{JAyB0hdb50xsE(Hsa1RBtY'
    'b3%!4)<vq4-(4$)O3_@VP<E`>MCsra<iIEnOn3;bTcOs$a4&|+);a8HEnL9)bI9-tVnLiveFwnlI5-8%rfFPZfGp;OQ`OmqvGaYa'
    '!AI_3EJk2TM;FMxeQI0UcV6Op)fgCEUGJ*GOuIrXS24oHn<+_SaLO_dT%ngoi}a>r{UZC@lJ9KDdVcGO)u?%jdD8^V2Lh_T0=?Ew'
    'Mw7$FDzMr@zBDdwDoe_!*I$nUB{aZH)gz^SJ=kne0*JdIG}R(OJtV)-G&w48rp>JPZaP;{feW+sT3k@NWg1ourq!9xPEV7YlxNyD'
    'pF&}Ha;$rMJ#7Kv1We`)X`H+@z;UvD+0&aT1P3N9iwVbDE`Na$%{U@gXd0O1MiH09z0U5W(m4f&I3ryQc70~Z!JvpxaOstm9&-H+'
    'uM0M7PJKAQPM{Kmtppn8TsU9XV<b_9DNkc@KI2(9@k3G*da}^AL6g(H{ip*?X9<+$DX271M8%3I^9clN%R!mTS&GP^VI`<0j1%Ny'
    'i7LCNa|NY9Y91V16;9-^EDhoc+&X|DZ(Gi%Px&OqhyZb8UH+`PtRUl+6~_rfWe>W{ho$h`l6@S7Pphl!>3m<CZp`>NWae~W!~Vqh'
    'O{r&&F8%^NsGF~1AoBgAL68{t*vv^K=iV4fT}ME^dC5d71ZlfZoCYs=3O4c$L))-nAqn!-AfWd0B#JsPP^3v}uca`yC42d%fS(OP'
    '6+m5a;=l?gKKoC)U2ZRV26#y;o+d8;9h7Us2s54k6k>q*%ob+iR?(ZjKZe>OU!-AF@v}uMN$k_GjNBQ)$!aOTpVh*q2p0k|!2wW5'
    'F?HRE7-R3I3Yecs?drAVnltvv7vBqrT_c5yA1O?+b<vWeLb!M@oD<oyn=GJsiLiM!7x5LaSrIc!5+|Z1vhptTC=<q31;B^pIfhyP'
    'e2L0}Q;oj#z87<&F7KSGD}tMEV)KZGPBSeV>SdR6h9tMSEeQ_s1QZvRO){}*#1SF-xt=~c->%ZqnOcG&0}L*x(4U}+ShT;0b(n}R'
    'Fe3KB6#tShUr2=uym6AC+obWds~-kAeot9RHsQ8dnru#hEPIFBeLYX#p%5oV=TdK*BC0NSPKpC}$sMnGp=#&QX%%-(IU-7Ec1g}Q'
    'Z+WYD%=40mU|~S2jvP=kgQz7KwA0q@Kb_7pq1ooF$JVhWo?m`A7J|w%xTqp%*>?}>rW5*(=q`;=^DB}<qF^YPLc@W%lwj^T<fm5?'
    'Yn-;UeWeO5xcevoDW=yF47H=|kP8w%0s|CN^5aM`uIgwFm6v}Y^qNCC&eh7J8;VO7`1LPCC$ahdFdUBggKo$+pAinBXhJ35FEJ?)'
    'N(W;%)AsmoYQbDSCT;l3<)@dqz>Awf*(UgAs|)EyHCrECPF%{MS2yieJlO2FU?|$}k|aE?Hx27KA5Rvk>KkSj%B7^0ydEB1$?RZF'
    '1c2%W=E@lni==N`&T@aNM4{<R)#hJ&qyGe{3;Y}YMLN?~5jtm^dPzhfxg8jUVe2Xe+9<n`iU8zxEh-07d$P%6lz@Jfn^KftOwlOc'
    '2D1_g=>CwLL5>Mx+-7`veR}j$-CTo6PRJ@$s~;x`LJ~rZier@Wqd~<)DQkOlCS5he5qeYtK|LvTT811JbVE0~BZb-694K&xVoI}K'
    '3A7qoE0fa9yNY#Hl^KJ`4;5<Lmt67RvHss~ej`r$A}1{%xUY&L{6wiK%06*ehXz3)Hv#D?*V-E@e3Ym@g)Q?+OiYYmYcA{h_vpoK'
    '8<X~8OD=l>T5I^(VJqKnD`81#bP$9Vi^iwBBG`W+xIINupZ6&7KYTC&pyF7mA{*?q{^84gg`a$3aFY!gC!ICrV2FYwXo|L#uXcor'
    ')oLs+PdARfhk6<OC~Ml1l~g9Yop{Y-3`JlV%3L^&D$l**Uh`v6TthWH0ZM>ywUZ~x=&FnmUnb)W*u-S#If9MkqN}+lwby(#(qaMD'
    'kl7MINz_7q;Z?LAZRy#?#~ZqGF`uIV(qLmDu*Jif4nva!I}?K`i2r%Qa;vBxuf`GeI)k9>p3N?WVuFJe7$dl2sJMHuyPjUy^(-r|'
    'i??jBo6EXzcZct=U#;YYQ|qbmg{quSfAK^L-9iq*#Xu4=ea2|T(;`*{V+6vd3S>RQA|A2qi2GnH;JsW!!Q4M%!f{Ydu60q~xT<2J'
    'j)gZpkBV2HCr=OpvQ1+&K@*!)@vlj|4X9&^1}9G8q-w;Z<fxJ!2Uf|*X3=BnZWoMtxToge=sz(!az+qBM?6f<o2ifszt<YR3;;}&'
    'k|C;kvYgSSn(ikFh@w<ILYiPGCfM^d>D;u`1pCsu`bN1ZLM(uw@Y1614eeRLXIoS2>Q8UDWm)RrpeCA*e%uL9@CU9$+o=gjSEgKd'
    'FtDj^j{-D#?tfElX<r|cyy|k*%w~u4^zxfR$jC&gK{=L|p5cYah1b{_fU`o`qzIT=N__fI$tNzu?~jik_lL`iB0$#UEaR?Msk5*v'
    '@hRyrUqk`Uvj;joT>mdL1N-3@q4WK1xQ?)x2WZoF+O}>E{2kJzZ?D8gsssj{yo0jn;|-mNeh_TAjDY&&xpujj%`8xWkxRNZ+h&BK'
    'fLx*~)YUI+H9u{EQ5gk)Ko{P!M7LX`d!aIPCRIJP?~OOzs8JwWzK;zPf4B;9is<i$p^zMg3X$T%G!Lv!DcSh*jnILk%Xh)H-+b6D'
    ';Sa-{cn?T!ctoF)!fIz@XV^S=I|+)nEjey0|HKujYE8T)$l49Z7)8Hx|A`ajvys40w)`3)BVgyyF!fK+0rTW_s#Btwl?>Qf+z4XL'
    '=T_my-K2=+i*5@N_v+4!97kE)v|aU=E|Uf~_wx9SNC?3;Jl*Yh8lmOwi8%u{zeHOXKF%nK$YVqM$+p}H-qQZ!458j($?;lsXZFqb'
    'gN(LgJ1pE&gZpZ^cEvw_MRSc^mLVZioZG8^au>EsB$%W-a$MU51XV|pC^)&8WcvX5k?(R9p83W-qhmy%%Iv>A_rCy|$!Z2>!Lob9'
    'Zu-;?BfB>`rrO;~XmtVE4A(!ghb?L*iENP#0Q}jMG1T6mWDxpI4P;?0-<grnzE>p8qNj6x-q1^{G4WC0bS4CqSP#1`)JM2}^n<V7'
    'ToQ64T*3@`l7DeilZXZKqUx@nZ}UTCjF&5qVTz+s&L2WMY<!cTDn+rpA>|zb9h(6)jUS~M=5S)kdYi4p<QD&(HM%4Gsas0_pI#|K'
    'acl-t|NHR|IL7}rq1?Zr--0@q1Zj}463e6R1fkD~F;9s5V)C3x7r*2M_+(COOlAM4Tdhdr;4BfX-24l$@99eyvgh}rO>Q&wbMOeE'
    '7`bmz!?JP3kWR2XaJCsnn}_tAIKO04n!dpi&g*@?h*38bl8@(Q&K!SXi|Q}K;=YXBHGl9%)!O{rDpw|Sh>rYCGz=+KxFAD=o;Tr7'
    'dSk_esi;}g@^mlbTVnh{d96Cc;nPm3HS}~B3%ge@4OlS5^h<`6_VKU?;0r>N2lr|N+f7gyb_}&)phvx${C;n-MO28au%l#rIQBw4'
    'TqqCZobt{6Ld|QRZus;pylhJ?pEj<ra8sd!TBC<8W6S`2;X^NNnf~Ck0P!?orGlR*0{<|ix~U_QxNsLuc&s=2h#P5zx(naoWrB0*'
    '4o$m%#f;5zN*(?4cy<;|YDtNmNF6zoT)06wS$p(hcl%h+S_PWKZwN9moapqt@fC^Uik3a;!Sqj9blkeny?4??_qJF59hTrDlTxhv'
    'Bn|EDVdE#wNp6#K4HAe<{?sHh-dtHd#6X0hVPmH{ywlep1X&%)z$bgYo1$hrF5#xe1Q+yW3B0k85)E{63$+LtK2kOkTlM5wclxS;'
    'N%8Le(SK}aKa-wNZhR|nz2yO`ysFrBizlOv;Eajncx*S(YkgUL;-s^~=-}61oqcSWy>dADRWr2P>=+2JiLTkgWHix94w4IEQ`q_m'
    'I||DNPQ~-0-NOxiPn6u|b>KW!NRC7$FC@cKz3$ub3)^yq{Nte-4`Um=z9(`#5Wrw#AFll$fPgk*yU@JE1`e8C)#vsX&<@1#+~n@?'
    'EoEfKNIZ8$tczm}MW2Bb^I)`a<;X2h4R~9gL~O-ttB<UqP#7nq4l2Rs;;~x%)QX<U)7{W?VvuPx(*uBbjRX)c-W2KBacA<x&Acp*'
    'S*hxQFak6`Den2&8(0A0t{c_BA5zqgoVoA*t9IgMa=xsA8E)amsjODZcgO~YTqs#%eq*26m}8~8n28?FI14R2#bkQH>hpC#Z;}?|'
    'cRYQ6U^%Q}>o#q9nKzqEwXvd%o7jMxVuo%5%og#)z8y)ne|UsqXP0`o*Vd+<>e6NdQl<^;t<`vLAaXcT)#2I{1;IpLtn#-rfIa-0'
    'qm|s;nZ+A=Cxogfkl)9<Q!%2xFfTB)+%w9X(Q7oe@`xo)i!!%AI-g~vmJ>Fw2gVKi@7xK90|JH(%8p8p?mu89pR=Za?mkP*=O}E1'
    '(8>9>)9dsng^DZ@n`~U0$L@1i`Oe<`U>G<qAg2*Vt|>Q|{bS+!OuGbAY7n=y+sNXFulWEie(VJR@1oxpSBB%EK~S7hKP+UHG*`pW'
    'OcBMO=9G4-maF+V<XRxSVyxm|HGsr%++mzmlHnQSmd3biWMZbNqimZ=$(<36TKRGh%Td*j36Sx5o2UHn>X;BpGzRzdLCf#g$~yu7'
    '?96$-19NP6{&z*41WXM9{eWS9S-V?~N@(OX3*(z$FGubN7G0@R6w;7txYS-HHN$DrCQ2cZ)bqV)VsP+66hrX+4lSBG)!B@}>a>v+'
    'cBuqole7**eQ@_PKi&!MlOzoNux;kwuA`IcQO{KGi#iBG2G^{<js4g|3+(yR&RpI5@3(ombi|em*?k*^yW2a;Dm5hH9N@#f0oJ6o'
    '0$CxO!#_@Gn0N#W5$%`da0oB!;Gn?{ko$TIO6A(L?On!VNpIM_64XI`VP?dIfphI75`~ucTDzsLm&!%GOTFY_ykua_nitY8bTI@J'
    'uzKy62<&=G*jW!66AcP^TmK15&_RR(E^j#WB=1F<*b6BE=9$4vf&0Xe=$6f2N)6xYf3%IYQ3eWF+$JmAiUX!eyz4BNuXm|=gzge_'
    'T?pBc|36s}qsah=cq28f64w+Y6BaZ3iy%HHy~DszXzeutB1_RX6AT`IeSmFEyHI{gzX_c8nFnul5pfvU86(P}1rs5DxUsB2?xCbN'
    'H%IvWPJC8}8_w%lDTSmKbr!{~J<E1V0~l^IS7T48VJ}tSqoq;fHuCUHpv(pBTz*Z&cqg;9g6HY%X>TzgR@C-Z97?xMmzo}r1K=u6'
    'DX&s0mkmOV&Q-i`s|&vgS>BI5lpw6gV4|{;11TLLLE)eS7~wSoWMO`M9eNra;yE}wfa)8&CPpQ<Qhm*QXVw4>GrY539teWshztk`'
    'gW_)6SmF)XG6|7UHADltot4G~@9dfgO{%QStFL?5?!6vX4iz5%B3+*MkRZxEs)fMX=$+nHJ}BsmiLO*|!}ekLJ$5@!>B~K~(#-k`'
    'r_OMx=PaoTpvEvviN8{tfR!}`%9F`I_bY9S>2;WJ=F27*t+!M*Uev9?QrRkk$;RT)&hJEW)N2*{YE&La2L+bmgb^$k$K0eZKDTOS'
    'a89nK`kvi%B;L-Z^F!R4>96$6is({j5hYk&&z4b=hbHrl?IQMS4b!Z;<D@|*3U#bqT`uw8pN_UVa;6PYw$D{|m0yGSlGKopE}13#'
    '+yO|vSEX0O5z2p;e?%;qR_z!1exI8)PUH*CaWaL#bWB^!E5p7L_pJz9OzGQgpH{O7+0t2@q1ECt^QYS0UA}avw9uKgj^!CU($|j|'
    '&c$~{6r&KIrB}|nG_CnxTWNluSTpZolMdQo-W6xDZH>sw9@o)qH|O<qK}XxW(*n@HxGBNSS^U!sh?7>rnA`-6Zl9g*rb6T^68~6x'
    '(zZRMYrSZ{_<&#tjHHR&DiqHyltvDD5958G(BeY`-{EHo{3kVhVbGnsE4!T?;Trn2e$u1PEnMnu8&JFN({iIJsQJexPX|qX%y9pT'
    'W;9GTL?xNrdL3<~?}=YZ&2AsgK@)(vUCa}K3J1+k{JBjtQ6Iv*e{2&evifYgk4ZYfU`7LA<ogUWP5Unzg-nwww`SJ6*6$SXCR6h$'
    '6>(myO+>Y|Z3^|}xXd?gcl}E&#JLdq{ROg?iElPV4b+zKn~(^dD(!>WS0#=Ic(XOnj>{(1>~eOyUYMnJ+sDA8Ugb{|TK~p3*c3wJ'
    '^JbP|%7xVlfK!T1LEfq)()c%Dw`#vGAQfif=e4Qe!XJZg&Z_(9RulVTn@2G7;NsT1z0X)JI?u4uNX{GG`eUDc7zxC4{SxtLwGT{w'
    'bo|gSJZH;zxw<-~E`$?m#eN)$1HQt1-DA*J5CjUxtS+R*wZKF#%RcvW$zU!hafM;$1ov}Rf0P^ukZ9@qzZpbfMrLM0xLm1bTV9(V'
    'w-yd$*~v1SRNYLkfjiZ=W2z}DVFySmJJaHiEy9;n?@A|%b3Zc=$LnQrEjW{X6JZ;ss*tKG%+D81i<y=W$3eizx0b`m<RYm9<?MnG'
    'rgI9U<q_pm5e+D>8A%h)Kjufr#9^C)=N0x5uB}YG7c(}4Om7!_B<!9XD)+-oZKVNzAGc`e$3d$d2>GkCIwnRfK1SxTd8{YW;}h*f'
    'egnxS+uCpKXx}H*0kNt_CN?g&Z)VUm{{fB2Ho9Y?meh*=hg#K+L1U4|p4Di^6-fu=(QQ^_NZQK5Dt&e1Uu(#DzZu>ibK#ywn5xOP'
    '>*qNSDZ|W^Qq07R813{#m5v)0L}S+rAs{80`Q2>1nu$fOGI;EI9gcn6k9qT1Sg1%==lai2KXR$rP&#xGDAZhJNW$lv&x~wSLck`;'
    '+~3cp=i1Hi%QSUeCc9)a06g!Vq6KIMgF9Wjg(&~hQ4D@pHC@Bb47YPgj&U$0B^gML?4g+4sDgNo7fxoH2Y_y&=<>shge&Q!7V-9)'
    'MZV~N_7eeq;%8O%Irfkjs6@f(4EGs=v3VW+{!GcrX4c$1_idU7j7DjRheO~rl34E|ayK5V(kBAy<tY9?Z=*Vxu`^7dosE@~Y+{%s'
    'F7jeKdhf4o+9WO?YxXF%h&POI%8?7X>0r?eL07}_;heGslQRm<D#}S)HYBw@QiL)M`jAvR5Vz?Jc-3~zAISJ2*nTst-WOj6;XrrC'
    ')BV?EMthX5S^w(fyNN{glSqqLdv!B@UcUN+Q^I5}sOrvC_jE~@FXZQRS$=m=5Z7^PGu_@6;{WzyAVSr)#WiJw{M~1shT(fdC2d%*'
    '!tAjB+~h>l?M6gUI(F7{-=gxW!yeP^6x2ZV=bdX|1*jNB>miL}>c=)<-!+oN{lhI&(L2p5!w6Se+djCSZJBK`Q^OkR!&s<|?dgyO'
    '$&O_7M>d_?IM6HDuWb}FWM2`HSp7E{e%XOq(5k*Dr>Ny??4ND-``4|3a{r_SV>rFu4$ok~>RO-v;KY0w7Du`<kyDD(uUtI7OUN)Z'
    'd}ITCAVrIVGfc#1><F^0KC#-4X)q3AZOOsj+sAB@^UK62cHED;rQlh{r@SqXmQ1&zJ)%E!?v9~pNtkonF`rT~4ebZp=8hy{8`}IA'
    'nQo@14BOT+bl+e9rfr$X7QwRMZ$1@fJEHURHP1guxBMGEh@Ds2_fRzH!@1xleZrxSo1WJc+(>{;-UrLA2(bZcS-kwaj79Zi^>!&I'
    'AFuNDi&of$`>k*PypvgrQKJFwV%h%&P=EL%6`9fB*Hs(S5bG&4rmM_6zJNg;{1UcP=8GYhS|7YHC#KND7qLzXMY7Bf=@lM`h3Y39'
    '{Bu{+-{2Aejw)cF`_**)gkx{DE5e`#uY?nWds$@w{qt;(5B4KLf7|>|-Oc8TwmT+_d@Ez<7iG*upO@XuFt*!R(pRM{*2XS0Yo<=S'
    '28TCIym@x!E2cvTws?r&LCfaDdAr%NEg55@4#eXnxfUqjXuN5M_xZz`*i98@2pTDI`eu6bZ3S+1{Di@G$c)3ieb?Sxhe$F7zO&UL'
    'nWh7DjBzJP*qw-d!Ep#l#{~Qf#}ppr?@FWtM!D>$h`mk0Fq_~zEF0;+pVJwbg~8vx*gkQRPF}g7S9RFI|95SoQL!s{*Y3FXrw>5b'
    '=#9V;`F?^FX+G_r`6vSE@n*zU?CTY!!@>UsiOeskH<{er)B(fOzBby+#jPP7AyG3)`>v-5>nG7PT8!tf7)h@D3gPkfivuzkR#ny?'
    'W59<@v~ioph4mZyQo8~6@MEFbooGCmmWf;D;j;<tCU)5osb;@_63lRsKizB_?mI6J``w|97f=rXvkgaiA9?!(qQ9#6xeuaxtO;;`'
    'JCui+LD_oTWTF6WxNSHI;k`6LwswE*j1F8ffB?*lCA-JNBr$!5ZgoFst5)!fa#D*SJ74?Ay%#Tg*>2@~^~I0!y_1hZ#N)#vpKiZA'
    '#oqyz`+pFu^b=nZfnYbJv%Dhb(`R2P$o%#?bTz|A;*9`37eVM9SMuGVIyl39?1K?(^Ub`E2TkS&m31sDH~DY59K>90GU=EFj%<hR'
    'Z5+Q=A+sdSUFk^j4C*^xBHrp6{NSC9nyc<ELL0~%<x)w8ocq@2R2%k-yca|2<_et+XsKZP6j#zhImeH9T13j7nPfNx!_MSTnw|Lu'
    '%Js;mJkmXw63RZMM}|s(9z@BhU)hW*bGgD!E`M?5xW!e*aolyfEVtf@?LVf3YIk+3+oI-f+Bq^m`LwtKL9=hPZIgSCunIk!ev7Pe'
    'ns?Cti+N~nqmC2%-FFy$+CD<GWi1~xv#=>tzKWeR;73cP8{5xffh)}er(d*Bj`g_!+YgbAfnF2fx3Gav^#1|yqszh'
)

SEEDS = (42, 43, 44)
SPLIT_SEED = 42
HIDDEN = (64, 32, 16, 8)
PRETRAIN_EPOCHS = 100
HEAD_EPOCHS = 30
FINETUNE_EPOCHS = 1500
PATIENCE = 200
LR = 0.001
MODES = ('scratch', 'ae', 'rbm')
RBM_LR = 0.05
RBM_BATCH = 32
RBM_DECAY = 0.0001


def load_dataset(name):
    if name == 'credit':
        raw = zlib.decompress(base64.b85decode(CREDIT_B85))
        if hashlib.sha256(raw).hexdigest() != CREDIT_SHA256:
            raise ValueError('Нарушена целостность встроенного crx.data.')
        data = np.array(list(csv.reader(io.StringIO(raw.decode('ascii')))), dtype=object)
        if data.shape != (690, 16) or not set(data[:, -1]) <= {'+', '-'}:
            raise ValueError('Неверный формат Credit Approval.')
        x, y = data[:, :-1].copy(), (data[:, -1] == '+').astype(int)
        missing = x == '?'
        x[missing] = np.nan
        numeric = [1, 2, 7, 10, 13, 14]
        for j in numeric:
            x[:, j] = x[:, j].astype(float)
        info = dict(name='Credit Approval', class_names=['−', '+'],
                    numeric=numeric, categorical=[j for j in range(15) if j not in numeric],
                    missing_by_feature=missing.sum(0).tolist(),
                    missing_rows=int(missing.any(1).sum()), sha256=CREDIT_SHA256)
    else:
        data = load_breast_cancer()
        x = data.data.copy()
        y = (data.target == 0).astype(int)  # Положительный класс M, отрицательный B.
        info = dict(name='WDBC', class_names=['B', 'M'], numeric=list(range(30)),
                    categorical=[], missing_by_feature=[0]*30, missing_rows=0)
    info.update(rows=len(x), raw_features=x.shape[1], class_counts=np.bincount(y).tolist())
    return x, y, info


def make_preprocessor(info):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', MinMaxScaler(clip=True))])
    transforms = [('numeric', numeric, info['numeric'])]
    if info['categorical']:
        categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        transforms.append(('categorical', categorical, info['categorical']))
    return ColumnTransformer(transforms, sparse_threshold=0)


def prepare_dataset(name):
    x, y, info = load_dataset(name)
    train, rest = train_test_split(np.arange(len(x)), test_size=.4,
                                  random_state=SPLIT_SEED, stratify=y)
    valid, test = train_test_split(rest, test_size=.5, random_state=SPLIT_SEED,
                                  stratify=y[rest])
    indices = {'train': train, 'valid': valid, 'test': test}
    processor = make_preprocessor(info)
    processor.fit(x[train])  # Ни validation, ни test не участвуют в fit.
    arrays = {part: np.asarray(processor.transform(x[ids]), dtype=np.float32)
              for part, ids in indices.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('После подготовки остались неконечные значения.')
    features = {part: torch.from_numpy(v) for part, v in arrays.items()}
    targets = {part: torch.tensor(y[ids], dtype=torch.float32) for part, ids in indices.items()}
    info.update(input_dim=arrays['train'].shape[1],
                split_sizes={part: len(ids) for part, ids in indices.items()},
                split_classes={part: np.bincount(y[ids], minlength=2).tolist() for part, ids in indices.items()},
                split_indices={part: ids.tolist() for part, ids in indices.items()},
                feature_names=processor.get_feature_names_out().tolist())
    return features, targets, info, processor


class Classifier(nn.Module):
    """Четыре скрытых слоя и выходной: всего пять обучаемых слоёв."""
    def __init__(self, input_dim):
        super().__init__()
        widths = (input_dim, *HIDDEN)
        self.hidden = nn.ModuleList([nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(HIDDEN[-1], 1)
        for layer in self.hidden:
            nn.init.normal_(layer.weight, mean=0., std=.1)
            nn.init.constant_(layer.bias, -1.)

    def encode(self, x):
        for layer in self.hidden:
            x = torch.sigmoid(layer(x))
        return x

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)  # Логит, без sigmoid.


def greedy_pretrain(model, xtrain, epochs, seed):
    """Алгоритм со слайда 13: отдельный автоэнкодер для каждого скрытого слоя.

    Метки и отложенные объекты намеренно отсутствуют в аргументах.
    Декодеры временные; ранее обученные слои фиксированы.
    """
    representation = xtrain.detach()
    logs = []
    original_head = copy.deepcopy(model.head.state_dict())
    for j, layer in enumerate(model.hidden):
        previous = [copy.deepcopy(a.state_dict()) for a in model.hidden[:j]]
        encoder = copy.deepcopy(layer)
        torch.manual_seed(seed + 1000 + j)
        decoder = nn.Linear(layer.out_features, layer.in_features)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=LR)
        history = []
        with torch.no_grad():
            initial = float((torch.sigmoid(decoder(torch.sigmoid(encoder(representation)))) - representation).square().mean())
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            reconstructed = torch.sigmoid(decoder(torch.sigmoid(encoder(representation))))
            loss = (reconstructed - representation).square().mean()
            if not torch.isfinite(loss):
                raise ArithmeticError('Неконечная MSE при предобучении.')
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                value = float((torch.sigmoid(decoder(torch.sigmoid(encoder(representation)))) - representation).square().mean())
            history.append(value)
        layer.load_state_dict(encoder.state_dict())
        # Проверяем, что обучение текущего автоэнкодера не изменило прошлые слои.
        for a, state in zip(model.hidden[:j], previous):
            assert all(torch.equal(a.state_dict()[key], v) for key, v in state.items())
        with torch.no_grad():
            representation = torch.sigmoid(layer(representation)).detach()
        logs.append(dict(layer=j+1, shape=[layer.in_features, layer.out_features, layer.in_features],
                         initial_mse=initial, final_mse=history[-1], history=history))
    assert all(torch.equal(model.head.state_dict()[k], v) for k, v in original_head.items())
    return logs


class BernoulliRBM:
    """RBM с бинарными узлами; CD-1 с выборкой скрытых и видимых состояний.

    W имеет форму (число видимых, число скрытых). Вероятностные входы
    [0, 1] стохастически бинаризуются перед каждым обновлением.
    """
    def __init__(self, encoder, seed):
        self.weight = encoder.weight.detach().T.clone()
        self.hidden_bias = encoder.bias.detach().clone()
        self.visible_bias = torch.zeros(encoder.in_features)
        self.rng = torch.Generator().manual_seed(seed)
        self.velocity_w = torch.zeros_like(self.weight)
        self.velocity_h = torch.zeros_like(self.hidden_bias)
        self.velocity_v = torch.zeros_like(self.visible_bias)

    def hidden_probability(self, visible):
        return torch.sigmoid(visible @ self.weight + self.hidden_bias)

    def visible_probability(self, hidden):
        return torch.sigmoid(hidden @ self.weight.T + self.visible_bias)

    def sample(self, probabilities):
        return torch.bernoulli(probabilities, generator=self.rng)

    @torch.no_grad()
    def cd1(self, probabilities, momentum):
        v0 = self.sample(probabilities)
        p0 = self.hidden_probability(v0)
        h0 = self.sample(p0)
        v1 = self.sample(self.visible_probability(h0))
        p1 = self.hidden_probability(v1)
        # Условные вероятности в статистиках уменьшают шум оценки.
        # Деление на фактический размер batch учитывает последний неполный batch.
        batch_size = len(v0)
        dw = (v0.T @ p0 - v1.T @ p1) / batch_size - RBM_DECAY * self.weight
        dv, dh = (v0-v1).mean(0), (p0-p1).mean(0)
        self.velocity_w.mul_(momentum).add_(dw, alpha=RBM_LR)
        self.velocity_v.mul_(momentum).add_(dv, alpha=RBM_LR)
        self.velocity_h.mul_(momentum).add_(dh, alpha=RBM_LR)
        self.weight.add_(self.velocity_w)
        self.visible_bias.add_(self.velocity_v)
        self.hidden_bias.add_(self.velocity_h)
        if not all(torch.isfinite(v).all() for v in (self.weight,self.visible_bias,self.hidden_bias)):
            raise ArithmeticError('Расходимость RBM.')

    @torch.no_grad()
    def reconstruction_mse(self, x):
        # Только диагностический mean-field проход; не целевая функция CD-1.
        return float((x-self.visible_probability(self.hidden_probability(x))).square().mean())

    @torch.no_grad()
    def fit(self, x, epochs):
        if not torch.isfinite(x).all() or torch.any((x<0)|(x>1)):
            raise ValueError('Входы Bernoulli RBM должны лежать в [0, 1].')
        initial = self.reconstruction_mse(x)
        history = []
        for epoch in range(1,epochs+1):
            order = torch.randperm(len(x),generator=self.rng)
            momentum = .5 if epoch <= 5 else .9
            for start in range(0,len(x),RBM_BATCH):
                self.cd1(x[order[start:start+RBM_BATCH]],momentum)
            history.append(self.reconstruction_mse(x))
        return dict(initial_mse=initial,final_mse=history[-1],history=history,
                    updates=epochs*((len(x)+RBM_BATCH-1)//RBM_BATCH))

    def transfer_to(self, encoder):
        with torch.no_grad():
            encoder.weight.copy_(self.weight.T)
            encoder.bias.copy_(self.hidden_bias)


def rbm_pretrain(model, xtrain, epochs, seed):
    """Стек d-64, 64-32, 32-16, 16-8. Метки и test не передаются."""
    representation=xtrain.detach()
    head_before=copy.deepcopy(model.head.state_dict())
    logs=[]
    for j,layer in enumerate(model.hidden):
        previous=[copy.deepcopy(a.state_dict()) for a in model.hidden[:j]]
        rbm=BernoulliRBM(layer,seed+2000+j)
        info=rbm.fit(representation,epochs)
        rbm.transfer_to(layer)
        with torch.no_grad():
            expected=rbm.hidden_probability(representation)
            actual=torch.sigmoid(layer(representation))
            assert torch.allclose(expected,actual,atol=1e-6)
            representation=actual.detach()
        for a,state in zip(model.hidden[:j],previous):
            assert all(torch.equal(a.state_dict()[k],v) for k,v in state.items())
        logs.append(dict(layer=j+1,shape=[layer.in_features,layer.out_features],**info))
    assert all(torch.equal(model.head.state_dict()[k],v) for k,v in head_before.items())
    return logs


def test_rbm():
    torch.manual_seed(7)
    layer=nn.Linear(3,2)
    rbm=BernoulliRBM(layer,17)
    x=torch.tensor([[0.,1.,0.],[1.,0.,1.],[.25,.5,.75]])
    w,bv,bh=rbm.weight.clone(),rbm.visible_bias.clone(),rbm.hidden_bias.clone()
    generator=torch.Generator().manual_seed(17)
    v0=torch.bernoulli(x,generator=generator)
    p0=torch.sigmoid(v0@w+bh)
    h0=torch.bernoulli(p0,generator=generator)
    v1=torch.bernoulli(torch.sigmoid(h0@w.T+bv),generator=generator)
    p1=torch.sigmoid(v1@w+bh)
    expected_w=w+RBM_LR*((v0.T@p0-v1.T@p1)/len(x)-RBM_DECAY*w)
    expected_bv=bv+RBM_LR*(v0-v1).mean(0)
    expected_bh=bh+RBM_LR*(p0-p1).mean(0)
    rbm.cd1(x,.5)
    assert torch.allclose(rbm.weight,expected_w,atol=1e-7)
    assert torch.allclose(rbm.visible_bias,expected_bv,atol=1e-7)
    assert torch.allclose(rbm.hidden_bias,expected_bh,atol=1e-7)
    rbm.transfer_to(layer)
    assert torch.allclose(rbm.hidden_probability(x),torch.sigmoid(layer(x)),atol=1e-7)
    bad=x.clone();bad[0,0]=1.1
    try:
        rbm.fit(bad,1)
    except ValueError:
        pass
    else:
        raise AssertionError('RBM приняла вход вне [0,1].')
    # Детерминизм CD-1 при фиксированном генераторе, включая неполный batch.
    a,b=BernoulliRBM(layer,19),BernoulliRBM(layer,19)
    a.fit(x,3);b.fit(x,3)
    assert torch.equal(a.weight,b.weight)
    print('RBM: формулы CD-1, ориентация весов, смещения, диапазон и воспроизводимость проверены.')



def train_supervised(model, features, targets, max_epochs):
    criterion = nn.BCEWithLogitsLoss()
    # Одинаковая supervised-схема для обоих режимов: сначала только head.
    hidden_before = copy.deepcopy(model.hidden.state_dict())
    for par in model.hidden.parameters():
        par.requires_grad_(False)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    model.train()
    for _ in range(HEAD_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(features['train']), targets['train'])
        loss.backward()
        optimizer.step()
    assert all(torch.equal(model.hidden.state_dict()[k], v) for k, v in hidden_before.items())
    for par in model.hidden.parameters():
        par.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    with torch.no_grad():
        best = float(criterion(model(features['valid']), targets['valid']))
    best_state, best_epoch, history = copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(features['train']), targets['train'])
        if not torch.isfinite(loss):
            raise ArithmeticError('Неконечная функция потерь классификатора.')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(criterion(model(features['train']), targets['train']))
            valid_loss = float(criterion(model(features['valid']), targets['valid']))
        history.append([epoch, train_loss, valid_loss])
        if valid_loss < best - 1e-7:
            best, best_epoch = valid_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch >= 300 and epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return dict(best_epoch=best_epoch, epochs_run=len(history), validation_bce=best, history=history)


def evaluate(model, x, y):
    with torch.no_grad():
        probability = torch.sigmoid(model(x)).numpy()
    truth = y.numpy().astype(int)
    prediction = (probability >= .5).astype(int)
    result = dict(accuracy=float(accuracy_score(truth, prediction)),
                  precision=float(precision_score(truth, prediction, zero_division=0)),
                  recall=float(recall_score(truth, prediction, zero_division=0)),
                  f1=float(f1_score(truth, prediction, zero_division=0)),
                  f1_macro=float(f1_score(truth, prediction, average='macro', zero_division=0)),
                  roc_auc=float(roc_auc_score(truth, probability)),
                  confusion=confusion_matrix(truth, prediction, labels=[0,1]).tolist())
    assert sum(map(sum, result['confusion'])) == len(y)
    return result, probability, prediction


def run(out, seeds, pre_epochs, fine_epochs):
    report = dict(settings=dict(seeds=seeds, split_seed=SPLIT_SEED, hidden=HIDDEN,
        pretrain_epochs=pre_epochs, head_epochs=HEAD_EPOCHS, finetune_epochs=fine_epochs,
        patience=PATIENCE, learning_rate=LR, threshold=.5, batch='full',
        dtype='float32', device='cpu', activation='sigmoid', scaling='minmax_train_clip', rbm_lr=RBM_LR, rbm_batch=RBM_BATCH, rbm_decay=RBM_DECAY), versions=dict(python=platform.python_version(),
        numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__), datasets={})
    import joblib
    for name in ('credit', 'wdbc'):
        features, targets, info, processor = prepare_dataset(name)
        joblib.dump(processor, out / f'{name}_preprocessor.joblib')
        dataset = dict(info=info, runs=[], summary={})
        for seed in seeds:
            torch.manual_seed(seed)
            base = Classifier(info['input_dim'])
            for mode in MODES:
                model = copy.deepcopy(base)
                assert all(torch.equal(model.state_dict()[key], v) for key, v in base.state_dict().items())
                started = time.perf_counter()
                pretraining = []
                if mode == 'ae':
                    pretraining = greedy_pretrain(model, features['train'], pre_epochs, seed)
                elif mode == 'rbm':
                    pretraining = rbm_pretrain(model, features['train'], pre_epochs, seed)
                pre_seconds = time.perf_counter() - started
                trained = train_supervised(model, features, targets, fine_epochs)
                total_seconds = time.perf_counter() - started
                # Test впервые используется после выбора весов по validation BCE.
                metrics, probability, prediction = evaluate(model, features['test'], targets['test'])
                row = dict(seed=seed, mode=mode, pretraining=pretraining,
                           pretrain_seconds=pre_seconds, total_seconds=total_seconds,
                           test=metrics, **trained)
                dataset['runs'].append(row)
                torch.save({'state_dict': model.state_dict(), 'input_dim': info['input_dim'],
                            'hidden': HIDDEN, 'positive_class': info['class_names'][1], 'activation': 'sigmoid'},
                           out/f'{name}_{mode}_{seed}.pt')
                np.savetxt(out/f'{name}_{mode}_{seed}_predictions.csv',
                    np.column_stack([info['split_indices']['test'], targets['test'].numpy(), probability, prediction]),
                    delimiter=',', header='row_index,true_class,p_positive,predicted_class', comments='')
                print(f'{name}, seed={seed}, {mode}: F1={metrics["f1"]:.4f}, '
                      f'accuracy={metrics["accuracy"]:.4f}, epoch={trained["best_epoch"]}', flush=True)
        for mode in MODES:
            rows = [a for a in dataset['runs'] if a['mode']==mode]
            dataset['summary'][mode] = {}
            for key in ('accuracy','precision','recall','f1','f1_macro','roc_auc'):
                v = [a['test'][key] for a in rows]
                dataset['summary'][mode][key] = dict(mean=float(np.mean(v)),
                        std=float(np.std(v, ddof=1)) if len(v)>1 else 0.)
            dataset['summary'][mode]['seconds_mean'] = float(np.mean([a['total_seconds'] for a in rows]))
        paired = [next(a['test']['f1'] for a in dataset['runs'] if a['seed']==seed and a['mode']=='rbm') -
                  next(a['test']['f1'] for a in dataset['runs'] if a['seed']==seed and a['mode']=='scratch') for seed in seeds]
        dataset['paired_rbm_minus_scratch_f1'] = paired
        report['datasets'][name] = dataset
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset','mode','seed','best_epoch','accuracy','precision','recall','f1','f1_macro','roc_auc','total_seconds'])
        for name, dataset in report['datasets'].items():
            for row in dataset['runs']:
                writer.writerow([name,row['mode'],row['seed'],row['best_epoch'],
                    *[row['test'][key] for key in ('accuracy','precision','recall','f1','f1_macro','roc_auc')],row['total_seconds']])
    return report


def plot(report, out, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    labels={'scratch':'Без предобучения','ae':'Автоэнкодеры','rbm':'Стек RBM'}
    colors={'scratch':'#2869a6','ae':'#c27232','rbm':'#39854a'}
    seed=report['settings']['seeds'][0]
    for name,dataset in report['datasets'].items():
        selected={mode:next(a for a in dataset['runs'] if a['seed']==seed and a['mode']==mode) for mode in MODES}
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for mode in MODES:
            history=np.array(selected[mode]['history'])
            for j,ax in enumerate(axes):
                ax.plot(history[:,0],history[:,j+1],color=colors[mode],label=labels[mode])
                ax.set(title=['Обучение','Валидация'][j],xlabel='Эпоха дообучения',ylabel='BCE')
                ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'{dataset["info"]["name"]}; seed = {seed}')
        fig.savefig(out/f'{name}_learning.png',dpi=180)
        fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
        maximum=max(np.array(selected[m]['test']['confusion']).max() for m in MODES)
        for ax,mode in zip(axes,MODES):
            cm=np.array(selected[mode]['test']['confusion'])
            ax.imshow(cm,cmap='Blues',vmin=0,vmax=maximum)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=15,
                            color='white' if cm[i,j]>.6*maximum else 'black')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=dataset['info']['class_names'],
                   yticklabels=dataset['info']['class_names'],title=labels[mode],
                   xlabel='Предсказанный класс',ylabel='Истинный класс')
        fig.suptitle(f'{dataset["info"]["name"]}; тест, seed = {seed}')
        fig.savefig(out/f'{name}_confusion.png',dpi=180)
        fig,axes=plt.subplots(2,2,figsize=(9,5.5),layout='constrained')
        for ax,stage in zip(axes.flat,selected['rbm']['pretraining']):
            ax.plot(np.arange(1,len(stage['history'])+1),stage['history'],color=colors['rbm'])
            ax.set(title='RBM '+'–'.join(map(str,stage['shape'])),xlabel='Эпоха',ylabel='Реконструкция MSE')
            ax.grid(alpha=.2)
        fig.suptitle(f'{dataset["info"]["name"]}; диагностическая ошибка RBM, seed = {seed}')
        fig.savefig(out/f'{name}_rbm.png',dpi=180)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for ax,(name,dataset) in zip(axes,report['datasets'].items()):
        means=[dataset['summary'][m]['f1']['mean'] for m in MODES]
        stds=[dataset['summary'][m]['f1']['std'] for m in MODES]
        ax.bar(np.arange(3),means,yerr=stds,capsize=4,color=[colors[m] for m in MODES])
        ax.set(xticks=np.arange(3),xticklabels=['Без ПО','AE','RBM'],ylim=(0,1.05),
               ylabel='F1 положительного класса',title=dataset['info']['name'])
        for i,v in enumerate(means):ax.text(i,.03,f'{v:.4f}',ha='center',color='white',fontsize=10)
    fig.suptitle('Среднее ± выборочное стандартное отклонение по инициализациям')
    fig.savefig(out/'comparison.png',dpi=180)
    if show:plt.show()
    plt.close('all')


def self_test():
    for name,counts in [('credit',[383,307]),('wdbc',[357,212])]:
        features,targets,info,processor=prepare_dataset(name)
        assert info['class_counts']==counts
        sets=[set(ids) for ids in info['split_indices'].values()]
        assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        assert len(set.union(*sets))==info['rows']
        scaler=processor.named_transformers_['numeric'].named_steps['scaler']
        assert scaler.n_samples_seen_==info['split_sizes']['train']
        assert all(torch.isfinite(a).all() and torch.all((a>=0)&(a<=1)) for a in features.values())
        # Изменение held-out объекта не меняет fitted-параметры preprocessing.
        before=scaler.data_min_.copy()
        raw,_,_=load_dataset(name)
        raw=raw[info['split_indices']['test']].copy()
        raw[0,info['numeric'][0]]=1e20
        altered=processor.transform(raw)
        assert np.array_equal(before,scaler.data_min_)
        assert np.isfinite(altered).all() and np.all((altered>=0)&(altered<=1))
    test_rbm()
    torch.manual_seed(42)
    original=Classifier(info['input_dim'])
    assert sum(isinstance(a,nn.Linear) for a in original.modules())==5
    a,b=copy.deepcopy(original),copy.deepcopy(original)
    logs=rbm_pretrain(a,features['train'],3,42)
    rbm_pretrain(b,features['train'],3,42)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert any(not torch.equal(x,y) for x,y in zip(a.hidden.parameters(),original.hidden.parameters()))
    ae=copy.deepcopy(original)
    ae_logs=greedy_pretrain(ae,features['train'],5,42)
    assert all(np.isfinite(v['final_mse']) for v in ae_logs)
    trained=train_supervised(a,features,targets,20)
    metrics,probability,prediction=evaluate(a,features['test'],targets['test'])
    assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
    tn,fp,fn,tp=np.array(metrics['confusion']).ravel()
    assert np.isclose(metrics['accuracy'],(tn+tp)/(tn+fp+fn+tp))
    expected=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    assert np.isclose(metrics['f1'],expected)
    print('Проверки пройдены: данные, [0,1], разбиение, отсутствие утечки, стек, перенос весов, метрики.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results_lab4')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    parser.add_argument('--pretrain-epochs',type=int,default=PRETRAIN_EPOCHS)
    parser.add_argument('--epochs',type=int,default=FINETUNE_EPOCHS)
    parser.add_argument('--no-show',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.pretrain_epochs<1 or args.epochs<1 or len(set(args.seeds))!=len(args.seeds):
        parser.error('Число эпох положительно; seed не должны повторяться.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.self_test:
            self_test();return
        args.out.mkdir(exist_ok=True,parents=True)
        report=run(args.out,args.seeds,args.pretrain_epochs,args.epochs)
        plot(report,args.out,not args.no_show)
    print(f'Результаты сохранены: {args.out.resolve()}')


if __name__=='__main__':
    main()
