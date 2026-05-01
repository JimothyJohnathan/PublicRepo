def pretrained_model_trainer(
        train_dir,
        test_dir,
        BATCH_SIZE,
        model_name,
        epochs,
        loss_fn,
        model_save_name,
        classes,
        lr,
        img_size=(288, 288),
):
    
    try:
        import matplotlib.pyplot as plt
        import torch
        import torchvision

        from torch import nn
        from torchvision import transforms, datasets, models
        from torch.utils.data import DataLoader
        from sklearn.metrics import accuracy_score

        from tqdm.auto import tqdm
        from pathlib import Path

    except:
        %pip install matplotlib, torch, torchvision, sklearn, scikit-learn, tqdm, Path

    num_classes = len(classes)

    model = models.get_model(f"{model_name}", weights="DEFAULT")

    def replace_head(model, num_classes):
    # --- 1. ResNet / RegNet style ---
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model

        # --- 2. Models with classifier ---
        if hasattr(model, "classifier"):
            classifier = model.classifier

            # Sequential classifier (EfficientNet, MobileNet, VGG, ConvNeXt)
            if isinstance(classifier, nn.Sequential):
                for i in reversed(range(len(classifier))):
                    if isinstance(classifier[i], nn.Linear):
                        in_features = classifier[i].in_features
                        classifier[i] = nn.Linear(in_features, num_classes)
                        return model

            # Single linear classifier
            elif isinstance(classifier, nn.Linear):
                in_features = classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
                return model

        # --- 3. timm models (many support this directly) ---
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes)
            return model

        # --- 4. Fallback: replace LAST Linear layer ---
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                in_features = module.in_features

                parent = model
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)

                setattr(parent, last, nn.Linear(in_features, num_classes))
                return model

        raise ValueError("Could not find a classifier head to replace.")
            
    model = replace_head(model, num_classes)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=lr)

    train_transform = models.get_model_weights(f"{model_name}").DEFAULT.transforms()

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE
    )

    print("Starting to train the model...")

    for epoch in range(1, epochs+1):
        for batch, (X, y) in enumerate(tqdm(train_dataloader)):
            model.train()

            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"Epoch: [{epoch}/{epochs}] Loss: {loss.item():.5f}")

            if batch % 300 == 0:
                torch.save(model.state_dict(), f"{model_save_name}")
                print(f"Saving Progress to ({model_save_name})...")

        if epoch == epochs:
            torch.save(model.state_dict(), f"{model_save_name}")
            print(f"End of training. Saving Progress to ({model_save_name})...")


    print("Starting to test on test data...")

    all_preds = []
    all_labels = []

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader):
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    accuracy = f"{acc * 100:.2f}%"

    return model, accuracy