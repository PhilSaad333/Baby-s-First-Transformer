import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from tqdm import tqdm



# claude wrote this to be more efficient with the tpu I'm using on colab (and gave me fancy progressbars)

def train_transformer_tpu(model, train_loader, val_loader, num_epochs, learning_rate):
    device = xm.xla_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    # Print dataset sizes for debugging
    xm.master_print(f"Train dataset size: {len(train_loader.dataset)}")
    xm.master_print(f"Validation dataset size: {len(val_loader.dataset)}")
    xm.master_print(f"Train batch size: {train_loader.batch_size}")
    xm.master_print(f"Validation batch size: {val_loader.batch_size}")

    for epoch in range(num_epochs):
        model.train()
        train_loader_tpu = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

        train_progress = tqdm(train_loader_tpu, desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                              leave=False, disable=not xm.is_master_ordinal())

        for batch in train_progress:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            xm.optimizer_step(optimizer)

            train_progress.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        total_val_loss = 0
        val_samples = 0
        val_loader_tpu = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)

        val_progress = tqdm(val_loader_tpu, desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                            leave=False, disable=not xm.is_master_ordinal())

        with torch.no_grad():
            for batch in val_progress:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                total_val_loss += loss.item() * inputs.size(0)  # Multiply by batch size
                val_samples += inputs.size(0)

                val_progress.set_postfix({"Val Loss": f"{loss.item():.4f}"})

        if val_samples == 0:
            xm.master_print("Warning: No validation samples were processed. Check your validation dataset and batch size.")
            avg_val_loss = float('inf')
        else:
            avg_val_loss = total_val_loss / val_samples

        xm.master_print(f"Epoch {epoch+1}/{num_epochs}, Avg Val Loss: {avg_val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model here if desired
        else:
            patience_counter += 1
            if patience_counter >= patience:
                xm.master_print(f"Early stopping triggered after epoch {epoch+1}")
                break

    return model