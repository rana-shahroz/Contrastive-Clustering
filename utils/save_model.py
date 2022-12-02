import os
import torch  


def save_current(args, model, optimizer, current_epoch) : 
    dir = os.path.join(args.model_path, f'checkpoint_{current_epoch}.tar')
    state = {'net' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : current_epoch}
    torch.save(state, dir)