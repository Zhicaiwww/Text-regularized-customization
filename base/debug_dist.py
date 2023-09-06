import torch
import torch.nn as nn
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import reduce , broadcast

class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):

        return (
            self.linear(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data
   
    def get_reg_loss(self,reg_vector = None):
        if reg_vector is None:
            if hasattr(self, 'fisher'):
                weight_reg = (self.lora_up.weight.pow(2) * self.fisher).sum()
            else: 
                # L2 norm of the weight matrix
                weight_reg = torch.norm(self.lora_up.weight, dim = [1], p = 2) + \
                         torch.norm(self.lora_down.weight, dim = [0], p = 2)
            return weight_reg
        else:
            lora_project_vector = self.lora_up(self.lora_down(reg_vector))
            return torch.norm(lora_project_vector, dim =[1,2], p = 2)
        
    def update_fisher(self):
        # update fisher information of lora-up matrix
        # TODO
        if not hasattr(self,'fisher'):
            self._fisher_steps = 1
            setattr(self,'fisher',torch.zeros_like(self.lora_up.weight))
        else:
            self._fisher_steps += 1
            _a = self._fisher_steps
            self.fisher = self.fisher*((_a-1)/_a) + self.lora_up.weight.grad.detach().pow(2)*(1/_a)

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)

if __name__ == '__main__':
    model = LoraInjectedLinear(20,40,r=4)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    ewc_prams = []
    freeze_prams = []
    for name, parameters in model.named_parameters():
        if 'lora_up' in name:
            ewc_prams.append(parameters)
        else:
            freeze_prams.append(parameters)
    for _ in range(10):
        model.zero_grad()
        in_features = torch.randn(1, 3, 10, 20).to(accelerator.device)
        out_features = model(in_features)
        likelihood = out_features.mean()
        accelerator.backward(likelihood, retain_graph=False)
        model.update_fisher()
        print(model.fisher)


        # accelerator.backward(likelihood, retain_graph=False)
