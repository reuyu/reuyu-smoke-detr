# C:\Users\user\anaconda3\envs\smoke_jetson\lib\site-packages\ultralytics\nn\smoke_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 이 파일이 필요한 모듈을 라이브러리에서 직접 가져옵니다.
from ultralytics.nn.modules.conv import Conv, RepConv
from ultralytics.nn.modules.block import C2f, HGBlock # 원본 HGBlock을 import (forward 상속용)

# -----------------------------------------------------------------
# 1. ECPConv, EMA (이전과 동일)
# -----------------------------------------------------------------

class ECPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ratio=0.25):
        super(ECPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = int(in_channels * ratio)
        self.rest_k = in_channels - self.k
        self.channel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            Conv(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        # RepConv는 'p'와 'g' 인자를 사용합니다.
        self.rep_conv = RepConv(self.k, self.k, kernel_size, stride, p=kernel_size//2, g=self.k)
        self.rest_conv = Conv(self.rest_k, self.rest_k, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        scores = self.channel_selector(x).squeeze()
        if not hasattr(self, 'fixed_indices'):
            scores_avg = scores.mean(dim=0)
            _, fixed_indices = torch.topk(scores_avg, self.in_channels)
            self.register_buffer('fixed_k_indices', fixed_indices[:self.k].sort()[0])
            self.register_buffer('fixed_rest_indices', fixed_indices[self.k:].sort()[0])
        selected_channels = torch.index_select(x, 1, self.fixed_k_indices)
        rest_channels = torch.index_select(x, 1, self.fixed_rest_indices)
        conv_result = self.rep_conv(selected_channels)
        rest_result = self.rest_conv(rest_channels)
        output = torch.empty_like(x)
        output.index_copy_(1, self.fixed_k_indices, conv_result)
        output.index_copy_(1, self.fixed_rest_indices, rest_result)
        return output

class EMA(nn.Module):
    def __init__(self, channels, groups=8):
        super(EMA, self).__init__()
        self.groups = groups
        self.c_per_g = channels // groups
        self.conv_1x1_h = nn.Conv2d(self.c_per_g, self.c_per_g, (1, 3), padding=(0, 1))
        self.conv_1x1_w = nn.Conv2d(self.c_per_g, self.c_per_g, (3, 1), padding=(1, 0))
        self.conv_3x3 = nn.Conv2d(self.c_per_g, self.c_per_g, 3, padding=1, groups=self.c_per_g)
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(groups, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_group = x.view(b * self.groups, self.c_per_g, h, w)
        x_h = F.adaptive_avg_pool2d(x_group, (h, 1))
        x_w = F.adaptive_avg_pool2d(x_group, (1, w))
        x_h = self.conv_1x1_h(x_h)
        x_w = self.conv_1x1_w(x_w)
        attn_1x1 = x_h + x_w
        attn_1x1 = self.sigmoid(attn_1x1)
        out_1x1 = x_group * attn_1x1
        out_3x3 = self.conv_3x3(x_group)
        fused = self.gn(out_1x1 + out_3x3)
        fused = self.sigmoid(fused)
        return fused.view(b, c, h, w) * x

# -----------------------------------------------------------------
# 2. ECPConv "드롭인" 교체 모듈
# -----------------------------------------------------------------

class ECPConvWrapper(nn.Module):
    """ HGBlock이 사용하는 Conv(c1, c2, k, act) 시그니처와
        호환되는 ECPConv + EMA 래퍼입니다. """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # ECPConv는 k=3, s=1을 가정하고 만듭니다.
        # k, s 인자는 호환성을 위해 받지만 ECPConv에는 3, 1을 사용합니다.
        self.ecp_conv = ECPConv(c1, c1, kernel_size=3, stride=1)
        self.pw_conv = Conv(c1, c2, 1, 1) # 채널 수를 c2로 변경
        self.ema = EMA(c2)
        
        self.shortcut = nn.Identity()
        if c1 != c2:
            self.shortcut = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.ema(self.pw_conv(self.ecp_conv(x)) + self.shortcut(x))

# -----------------------------------------------------------------
# 3. ECP_HGBlock (HGBlock 대체 모듈)
# -----------------------------------------------------------------

class ECP_HGBlock(HGBlock):
    """ 원본 HGBlock을 상속하되, 내부의 Conv/LightConv를 ECPConvWrapper로 교체합니다. """
    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False, # 이 인자는 무시됩니다.
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        # 부모 클래스(HGBlock)의 init을 호출하지 않고, 필요한 속성만 직접 초기화합니다.
        super(HGBlock, self).__init__() # nn.Module의 init을 호출
        
        # block = LightConv if lightconv else Conv (원본 코드)
        block = ECPConvWrapper # <-- 우리의 교체용 블록

        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        
        # Squeeze/Excitation 부분은 원본과 동일하게 초기화합니다.
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    # forward 메소드는 부모 클래스(HGBlock)의 것을 그대로 사용합니다 (상속받았기 때문).


# -----------------------------------------------------------------
# 4. MFFPN (특징 융합 넥 모듈) - (이 파일에 함께 둡니다)
# -----------------------------------------------------------------

class RCM(nn.Module):
    def __init__(self, channels, k_size=7):
        super(RCM, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.strip_conv_h = nn.Conv2d(channels, channels, (1, k_size), padding=(0, k_size // 2))
        self.strip_conv_w = nn.Conv2d(channels, channels, (k_size, 1), padding=(k_size // 2, 0))
        self.bn = nn.BatchNorm2d(channels)
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.mlp = nn.Sequential(
            Conv(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            Conv(channels // 4, channels, 1)
        )
        self.bn_mlp = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        y_h = self.pool_h(x)
        y_w = self.pool_w(x)
        y = y_h + y_w
        attn_map = F.sigmoid(self.strip_conv_w(F.relu(self.bn(self.strip_conv_h(y)))))
        fused_features = self.dw_conv(x) * attn_map
        out = self.bn_mlp(self.mlp(fused_features)) + identity
        return out

class MFFPN(nn.Module):
    def __init__(self, in_channels): # in_channels: [c3, c4, c5]
        super().__init__()
        c3, c4, c5 = in_channels
        self.rcm_fused = nn.Sequential(RCM(c3 + c4 + c5), RCM(c3 + c4 + c5), RCM(c3 + c4 + c5))
        self.pool_s4 = nn.AvgPool2d(2, 2)
        self.pool_s5 = nn.AvgPool2d(4, 4)
        self.rcm_s3 = RCM(c3)
        self.rcm_s4 = RCM(c4)
        self.rcm_f5 = RCM(c5)
        self.rep_c3_p4 = C2f(c4 + c3, c4, n=1, shortcut=False)
        self.rep_c3_p5 = C2f(c5 + c4, c5, n=1, shortcut=False)
        self.conv_p3 = Conv(c3, c3, 1)
        self.conv_p4 = Conv(c4, c4, 1)
        self.conv_p5 = Conv(c5, c5, 1)
        self.downsample_c1 = nn.Conv2d(c3, c3, 3, stride=2, padding=1)
        self.downsample_c2 = nn.Conv2d(c4, c4, 3, stride=2, padding=1)

    def forward(self, inputs): # inputs: [s3, s4, f5]
        s3, s4, f5 = inputs
        s3_pooled = self.pool_s4(s3)
        s4_pooled = s4
        f5_pooled = F.interpolate(f5, size=s4.shape[2:], mode='bilinear')
        fused_features = torch.cat([s3_pooled, s4_pooled, f5_pooled], dim=1)
        fused_context = self.rcm_fused(fused_features)
        s3_prime_p, s4_prime_p, f5_prime_p = torch.split(fused_context, [s3.shape[1], s4.shape[1], f5.shape[1]], dim=1)
        p5 = self.rcm_f5(f5) * F.interpolate(f5_prime_p, size=f5.shape[2:], mode='bilinear')
        p4_input = F.interpolate(p5, size=s4.shape[2:], mode='bilinear') + self.rcm_s4(s4)
        p4 = p4_input * F.interpolate(s4_prime_p, size=s4.shape[2:], mode='bilinear')
        p3_input = F.interpolate(p4, size=s3.shape[2:], mode='bilinear') + self.rcm_s3(s3)
        p3 = p3_input * F.interpolate(s3_prime_p, size=s3.shape[2:], mode='bilinear')
        c1 = self.conv_p3(p3)
        c1_down = self.downsample_c1(c1)
        c2_in = torch.cat([c1_down, p4], dim=1)
        c2 = self.conv_p4(self.rep_c3_p4(c2_in))
        c2_down = self.downsample_c2(c2)
        c3_in = torch.cat([c2_down, p5], dim=1)
        c3 = self.conv_p5(self.rep_c3_p5(c3_in))
        return [c1, c2, c3]
    ```

---

#### 2단계: `tasks.py` 수정하기

* **경로:** `C:\Users\user\anaconda3\envs\smoke_jetson\lib\site-packages\ultralytics\nn\tasks.py`
* **작업 1 (Import):** 파일 상단에 `smoke_modules`의 클래스들과 `HGBlock`을 import합니다.

    ```python
    ...
    from ultralytics.nn.modules import ( 
        AIFI,
        ...
        HGBlock,      # <-- 원본 HGBlock import
        HGStem,
        ...
        v10Detect,
    )
    from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
    
    # === ⬇️ 이 1줄을 추가하세요 ⬇️ ===
    from ultralytics.nn.smoke_modules import ECP_HGBlock, MFFPN 
    # ==========================
    
    from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
    ...
    ```

* **작업 2 (`parse_model` 수정):** `parse_model` 함수 내부(약 1045번째 줄)를 수정하여 `HGBlock` 클래스를 `ECP_HGBlock` 클래스로 덮어씁니다.

    ```python
    ...
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        
        # === ⬇️ 이 코드를 여기에 추가하세요 ⬇️ ===
        # m이 'HGBlock' 클래스 자체인지 확인하고, ECP_HGBlock으로 덮어씁니다.
        if m is HGBlock:
            m = ECP_HGBlock  # Swap the class
            LOGGER.info(f"SMOKE-DETR: Swapping HGBlock with ECP_HGBlock at layer {i}")
        # ===============================
        
        for j, a in enumerate(args):
        ...
    ```

---

#### 3단계: `smoke-detr-l.yaml` 파일 생성하기 (넥 교체)

`rtdetr-l.pt`는 `HGBlock`을 사용하므로 백본은 교체되었습니다. 이제 **넥(Neck)**을 `MFFPN`으로 교체하기 위해 `smoke-detr-l.yaml` 파일을 만듭니다.

* **파일 생성 위치:** `C:\Users\user\Documents\Projects\`
* **파일 이름:** `smoke-detr-l.yaml`

* **파일 내용:** (원본 `rtdetr-l.yaml`의 백본은 그대로 두고, **Head** 부분만 `MFFPN`으로 수정한 YAML입니다.)

    ```yaml
    # C:\Users\user\Documents\Projects\smoke-detr-l.yaml
    
    # 파라미터
    nc: 1  # 클래스 수 (연기 1개)
    scales:
      l: [1.0, 1.0, 1024] # RT-DETR-Large 스케일
    
    # 백본 (rtdetr-l.yaml 원본과 동일)
    # HGBlock은 tasks.py에서 ECP_HGBlock으로 자동 교체됩니다.
    backbone:
      # [from, repeats, module, args]
      - [-1, 1, HGStem, [3, 32, 48]]  # 0
      - [-1, 6, HGBlock, [48, 48, 128, 3, 6]]  # 1
      - [-1, 1, DWConv, [128, 128, 3, 2, 1, False]]  # 2 (s3 피처맵 추출)
      - [-1, 6, HGBlock, [128, 96, 512, 3, 6]]  # 3
      - [-1, 1, DWConv, [512, 512, 3, 2, 1, False]]  # 4 (s4 피처맵 추출)
      - [-1, 6, HGBlock, [512, 192, 1024, 5, 6, True, False]]  # 5
      - [-1, 6, HGBlock, [1024, 192, 1024, 5, 6, True, True]]  # 6
      - [-1, 6, HGBlock, [1024, 192, 1024, 5, 6, True, True]]  # 7
      - [-1, 1, DWConv, [1024, 1024, 3, 2, 1, False]]  # 8 (s5 피처맵 추출)
      - [-1, 6, HGBlock, [1024, 384, 2048, 5, 6, True, False]]  # 9
    
    # 헤드 (넥 + 디코더)
    head:
      # 10: s5를 256 채널로 축소
      - [9, 1, Conv, [2048, 256, 1, 1, None, 1, 1, False]]
      # 11: AIFI (논문 참조)
      - [-1, 1, AIFI, [1024, 8]]  
      # 12: AIFI 출력을 256 채널로 축소 (f5 출력) -> MFFPN 입력 3
      - [-1, 1, Conv, [256, 256, 1, 1]]  
      
      # (원본 RT-DETR-L 넥 부분은 모두 삭제)
      
      # 13: s3 피처맵 (2번 레이어) -> MFFPN 입력 1
      - [2, 1, Conv, [128, 256, 1, 1]] # 128 -> 256
      # 14: s4 피처맵 (4번 레이어) -> MFFPN 입력 2
      - [4, 1, Conv, [512, 256, 1, 1]] # 512 -> 256

      # 15: ★★★ 우리의 커스텀 MFFPN 넥(Neck) ★★★
      # 입력(from): [s3(13), s4(14), f5(12)]
      # in_channels: [256, 256, 256]
      - [[13, 14, 12], 1, MFFPN, [[256, 256, 256]]]
          
      # 16: ★★★ 최종 RTDETRDecoder 헤드 ★★★
      - [15, 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 6, 1024, 0.0, 'relu', -1, 100, 0.5, 1.0, False]]
    ```

---

#### 4단계: `train_teacher.py` 수정하기

`rtdetr-l.pt` 가중치를 `smoke-detr-l.yaml` 아키텍처에 로드합니다.

* **파일:** `C:\Users\user\Documents\Projects\train_teacher.py`
* **내용:**

    ```python
    # C:\Users\user\Documents\Projects\train_teacher.py
    
    from ultralytics import RTDETR
    
    def main():
        # 1. 'smoke-detr-l.yaml' 아키텍처를 로드합니다.
        model = RTDETR('smoke-detr-l.yaml') 
    
        # 2. 데이터셋 경로 설정
        data_yaml_path = r'C:\Users\user\Documents\Projects\ultralytics\smoke_dataset.yaml' 
        
        print("Smoke-DETR (Teacher) 모델 학습을 시작합니다...")
    
        # 3. 학습 시작
        results = model.train(
            data=data_yaml_path,
            
            # 4. 'rtdetr-l.pt' (Large 모델) 가중치를 로드합니다.
            #    백본(ECP_HGBlock)과 넥(MFFPN)은 이름이 달라 로드되지 않지만, 
            #    AIFI와 RTDETRDecoder 헤드 부분은 이름이 같아 로드됩니다. (Transfer Learning)
            weights='rtdetr-l.pt',  
            
            epochs=200,     
            batch=4,        
            imgsz=640,      
            optimizer='AdamW', 
            lr0=0.0001,        
            momentum=0.9,      
            weight_decay=0.0001, 
            project=r'C:\Users\user\Documents\Projects\smoke_jetson_project', 
            name='teacher_l_model_final', # 새 버전
        )
        
        print("Teacher 모델 학습 완료!")
        print(f"최종 모델은 {model.trainer.best} 에 저장되었습니다.")
    
    if __name__ == '__main__':
        main()
    ```

---

이것이 `rtdetr-r18.pt` 파일 없이 진행할 수 있는 가장 확실한 방법입니다.

1.  라이브러리를 초기화했습니다.
2.  `nn` 폴더에 `smoke_modules.py` **단일 파일**을 생성하고 **모든 커스텀 코드**를 넣었습니다. (ECP_HGBlock 포함)
3.  `nn/tasks.py` 파일을 수정하여 `ECP_HGBlock`, `MFFPN`, `HGBlock`을 **import**하고, `parse_model` 함수가 `HGBlock` 클래스를 `ECP_HGBlock` 클래스로 **바꿔치기**하도록 했습니다.
4.  `Projects` 폴더에 `HGBlock` 백본과 `MFFPN` 넥을 사용하는 **`smoke-detr-l.yaml`**을 새로 만들었습니다.
5.  `train_teacher.py`가 `smoke-detr-l.yaml`을 로드하고, 가중치는 `rtdetr-l.pt`에서 가져오도록 수정했습니다.

이제 **데이터셋을 준비한 후**, `C:\Users\user\Documents\Projects` 폴더에서 `python train_teacher.py`를 실행해 주세요.
