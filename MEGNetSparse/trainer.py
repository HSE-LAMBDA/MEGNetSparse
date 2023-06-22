import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from .model import MEGNet
from .utils import Scaler
from .struct2graph import FlattenGaussianDistanceConverter, GaussianDistanceConverter, AtomFeaturesExtractor, \
    SimpleCrystalConverter


def set_attr(structure, attr, name):
    setattr(structure, name, attr)
    return structure


class MEGNetTrainer:
    def __init__(
            self,
            config,
            device,
    ):
        self.config = config
        self.device = device

        if self.config["model"]["add_z_bond_coord"]:
            bond_converter = FlattenGaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        else:
            bond_converter = GaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        atom_converter = AtomFeaturesExtractor(self.config["model"]["atom_features"])
        self.converter = SimpleCrystalConverter(
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=self.config["model"]["cutoff"],
            add_z_bond_coord=self.config["model"]["add_z_bond_coord"],
            add_eos_features=(use_eos := self.config["model"].get("add_eos_features", False)),
        )
        self.scaler = Scaler()

        self.model = MEGNet(
            edge_input_shape=bond_converter.get_shape(eos=use_eos),
            node_input_shape=atom_converter.get_shape(),
            embedding_size=self.config['model']['embedding_size'],
            n_blocks=self.config['model']['nblocks'],
            state_input_shape=self.config["model"]["state_input_shape"],
            vertex_aggregation=self.config["model"]["vertex_aggregation"],
            global_aggregation=self.config["model"]["global_aggregation"],
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
        )

        if self.config["optim"]["scheduler"].lower() == "ReduceLROnPlateau".lower():
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.config["optim"]["factor"],
                patience=self.config["optim"]["patience"],
                threshold=self.config["optim"]["threshold"],
                min_lr=self.config["optim"]["min_lr"],
                verbose=True,
            )
        else:
            raise ValueError("Unknown optimizer")

    def prepare_data(
            self,
            train_data,
            train_targets,
            test_data,
            test_targets,
            target_name,
    ):
        print('adding targets to data')
        train_data = [set_attr(s, y, 'y') for s, y in zip(train_data, train_targets)]
        test_data = [set_attr(s, y, 'y') for s, y in zip(test_data, test_targets)]

        print("converting data")
        self.train_structures = Parallel(n_jobs=-1)(
            delayed(self.converter.convert)(s) for s in tqdm(train_data))
        self.test_structures = Parallel(n_jobs=-1)(
            delayed(self.converter.convert)(s) for s in tqdm(test_data))
        self.scaler.fit(self.train_structures)

        self.trainloader = DataLoader(
            self.train_structures,
            batch_size=self.config["model"]["train_batch_size"],
            shuffle=True,
            num_workers=0,
        )

        self.testloader = DataLoader(
            self.test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
            num_workers=0
        )

        self.target_name = target_name

    def train_one_epoch(self):
        print('target:', self.target_name, 'device:', self.device)

        mses = []
        maes = []

        self.model.train(True)
        for i, batch in enumerate(self.trainloader):
            batch = batch.to(self.device)
            preds = self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
            ).squeeze()

            loss = F.mse_loss(
                self.scaler.transform(batch.y),
                preds,
                reduction='mean'
            )
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            mses.append(loss.to("cpu").data.numpy())
            maes.append(
                F.l1_loss(
                    self.scaler.inverse_transform(preds),
                    batch.y,
                    reduction='sum'
                ).to('cpu').data.numpy()
            )

            train_mae = sum(maes) / len(self.train_structures)
            self.scheduler.step(train_mae)
            train_mse = np.mean(mses)

            return train_mae, train_mse

    def evaluate_on_test(self):
        total = []
        results = []
        self.model.train(False)
        with torch.no_grad():
            for batch in self.testloader:
                batch = batch.to(self.device)

                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                ).squeeze()

                total.append(
                    F.l1_loss(
                        self.scaler.inverse_transform(preds),
                        batch.y,
                        reduction='sum'
                    ).to('cpu').data.numpy()
                )
                results.append(self.scaler.inverse_transform(preds))

            cur_test_loss = sum(total) / len(self.test_structures)
        return cur_test_loss, torch.concat(results).to('cpu').data.reshape(-1, 1)

    def predict_structures(self, structures_list):
        print("converting data")
        structures = Parallel(n_jobs=-1)(
            delayed(self.converter.convert)(s) for s in tqdm(structures_list))
        loader = DataLoader(structures, batch_size=50, shuffle=False)
        results = []
        self.model.train(False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                )
                results.append(self.scaler.inverse_transform(preds))

        return torch.concat(results).to('cpu').data.reshape(-1, 1)

    def save(self, path):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(state_dict, str(path) + '/checkpoint.pth')

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])