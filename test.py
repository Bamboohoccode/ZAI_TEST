import sys
import nemo.collections.asr as nemo_asr

audio = sys.argv[1:] or ['/home/trinhkhng/voice-agent-pipeline/Dual_ChunkFormer/a.wav']
NEMO = "/home/trinhkhng/.cache/torch/NeMo/NeMo_3.0.0/hf_hub_cache/nvidia/parakeet-ctc-0.6b-Vietnamese/e27e90b93cec28e95955af8f0db08f79/parakeet-ctc-0.6b-vi.nemo"
asr_model = nemo_asr.models.ASRModel.restore_from(NEMO)
for path, out in zip(audio, asr_model.transcribe(audio)):
    print(path, '->', out.text)
