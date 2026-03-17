import torch

def federated_average(models):

    avg=models[0]

    for k in avg.state_dict():

        avg.state_dict()[k].copy_(

            torch.stack(
                [m.state_dict()[k] for m in models]
            ).mean(0)

        )

    return avg


def federated_train(global_model,loaders,device,rounds=3):

    for r in range(rounds):

        local_models=[]

        for loader in loaders:

            model=type(global_model)().to(device)

            model.load_state_dict(global_model.state_dict())

            opt=torch.optim.Adam(model.parameters(),1e-4)

            for X,y in loader:

                X,y=X.to(device),y.to(device)

                opt.zero_grad()

                out=model(X)

                loss=torch.nn.functional.cross_entropy(out,y)

                loss.backward()

                opt.step()

            local_models.append(model)

        global_model=federated_average(local_models)

    return global_model