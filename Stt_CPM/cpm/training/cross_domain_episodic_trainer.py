import argparse
import importlib.util
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from cpm.ccsa import select_episode_category_semantics
from cpm.cra import CategoryRelationalAlignmentLoss
from cpm.data import (
    EpisodicClassificationTask,
    build_episode_data_loader,
    load_source_meta_training_data,
    load_standardized_mat_dataset,
    prepare_target_domain_data,
)
from cpm.evaluation import evaluate_target_domain
from cpm.lcp import (
    build_class_identity_vectors,
    generate_label_conditioned_priors,
)
from cpm.runtime import (
    configure_experiment_logging,
    initialize_trainable_modules,
    log_model_parameter_counts,
    resolve_torch_device,
    set_reproducible_seed,
)
from cpm.training.model_factory import CPMModelBundle, build_cpm_models
from cpm.training.optimizer_factory import (
    build_cpm_optimizers,
    step_optimizers,
    zero_optimizer_gradients,
)
from cpm.training.prototypical_classification import (
    average_support_features,
    l2_normalize,
    negative_squared_euclidean_logits,
)
from cpm.visualization import save_classification_map


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location(
        "cpm_experiment_config",
        config_path,
    )
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"Cannot load config file: {config_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def build_episode_real_class_ids(
    real_class_ids: Iterable[int],
) -> List[int]:
    if torch.is_tensor(real_class_ids):
        real_class_ids = real_class_ids.detach().cpu().tolist()

    ordered_unique_ids = []
    seen = set()
    for class_id in real_class_ids:
        class_id = int(class_id)
        if class_id not in seen:
            seen.add(class_id)
            ordered_unique_ids.append(class_id)

    return ordered_unique_ids


def run_cross_domain_training_episode(
    source_task: EpisodicClassificationTask,
    target_task: EpisodicClassificationTask,
    models: CPMModelBundle,
    cross_entropy_loss: nn.Module,
    cra_loss_function: CategoryRelationalAlignmentLoss,
    encoded_source_category_names,
    encoded_target_category_names,
    config,
    device: torch.device,
):
    support_count = config["episode_support_samples_per_class"]
    query_count = config["episode_query_samples_per_class"]

    source_support_loader = build_episode_data_loader(
        source_task,
        samples_per_class=support_count,
        split="support",
        shuffle=False,
    )
    source_query_loader = build_episode_data_loader(
        source_task,
        samples_per_class=query_count,
        split="query",
        shuffle=False,
    )
    target_support_loader = build_episode_data_loader(
        target_task,
        samples_per_class=support_count,
        split="support",
        shuffle=False,
    )
    target_query_loader = build_episode_data_loader(
        target_task,
        samples_per_class=query_count,
        split="query",
        shuffle=False,
    )

    source_support, _ = next(iter(source_support_loader))
    source_query, source_query_labels = next(iter(source_query_loader))
    target_support, _ = next(iter(target_support_loader))
    target_query, target_query_labels = next(iter(target_query_loader))

    source_episode_class_ids = build_episode_real_class_ids(
        source_task.support_real_class_ids
    )
    target_episode_class_ids = build_episode_real_class_ids(
        target_task.support_real_class_ids
    )

    source_class_identity = build_class_identity_vectors(
        source_episode_class_ids,
        prior_dimension=config["label_prior_dimension"],
        device=device,
    )
    target_class_identity = build_class_identity_vectors(
        target_episode_class_ids,
        prior_dimension=config["label_prior_dimension"],
        device=device,
    )

    source_label_priors = generate_label_conditioned_priors(
        source_class_identity,
        noise_std=config["lcp_noise_std"],
    )
    target_label_priors = generate_label_conditioned_priors(
        target_class_identity,
        noise_std=config["lcp_noise_std"],
    )

    source_category_semantics = models.semantic_aggregation(
        encoded_source_category_names
    )
    target_category_semantics = models.semantic_aggregation(
        encoded_target_category_names
    )

    source_semantic_support = select_episode_category_semantics(
        source_category_semantics,
        source_episode_class_ids,
    )
    target_semantic_support = select_episode_category_semantics(
        target_category_semantics,
        target_episode_class_ids,
    )

    source_support_features = models.visual_encoder(
        models.source_spectral_mapping(source_support.to(device))
    )
    source_query_features = models.visual_encoder(
        models.source_spectral_mapping(source_query.to(device))
    )
    target_support_features = models.visual_encoder(
        models.target_spectral_mapping(target_support.to(device))
    )
    target_query_features = models.visual_encoder(
        models.target_spectral_mapping(target_query.to(device))
    )

    source_semantic_features = models.semantic_projection(
        source_semantic_support
    )
    target_semantic_features = models.semantic_projection(
        target_semantic_support
    )

    source_prototypes = average_support_features(
        source_support_features,
        class_count=config["target_class_count"],
        support_samples_per_class=support_count,
    )
    target_prototypes = average_support_features(
        target_support_features,
        class_count=config["target_class_count"],
        support_samples_per_class=support_count,
    )

    normalized_source_prototypes = l2_normalize(
        source_prototypes
    )
    normalized_target_prototypes = l2_normalize(
        target_prototypes
    )
    normalized_source_queries = l2_normalize(
        source_query_features
    )
    normalized_target_queries = l2_normalize(
        target_query_features
    )

    source_conditioned_prototypes = torch.cat(
        [normalized_source_prototypes, source_label_priors],
        dim=1,
    )
    target_conditioned_prototypes = torch.cat(
        [normalized_target_prototypes, target_label_priors],
        dim=1,
    )

    refined_source_prototypes = models.prototype_refinement(
        source_conditioned_prototypes
    )
    refined_target_prototypes = models.prototype_refinement(
        target_conditioned_prototypes
    )

    source_logits = negative_squared_euclidean_logits(
        normalized_source_queries,
        refined_source_prototypes,
    )
    target_logits = negative_squared_euclidean_logits(
        normalized_target_queries,
        refined_target_prototypes,
    )

    source_query_labels_device = source_query_labels.long().to(
        device
    )
    target_query_labels_device = target_query_labels.long().to(
        device
    )

    source_classification_loss = cross_entropy_loss(
        source_logits,
        source_query_labels_device,
    )
    target_classification_loss = cross_entropy_loss(
        target_logits,
        target_query_labels_device,
    )
    classification_loss = (
        source_classification_loss
        + target_classification_loss
    )

    category_relational_alignment_loss = (
        cra_loss_function(
            source_semantic_features,
            source_prototypes,
        )
        + cra_loss_function(
            target_semantic_features,
            target_prototypes,
        )
    )

    total_loss = (
        classification_loss
        + config["cra_loss_weight"]
        * category_relational_alignment_loss
    )

    source_hits = torch.sum(
        torch.argmax(source_logits, dim=1).cpu()
        == source_query_labels
    ).item()
    target_hits = torch.sum(
        torch.argmax(target_logits, dim=1).cpu()
        == target_query_labels
    ).item()

    episode_statistics = {
        "classification_loss": classification_loss.detach(),
        "cra_loss": category_relational_alignment_loss.detach(),
        "total_loss": total_loss.detach(),
        "source_hits": source_hits,
        "source_count": source_query.shape[0],
        "target_hits": target_hits,
        "target_count": target_query.shape[0],
    }
    return total_loss, episode_statistics


def build_argument_parser(
    default_config_path: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conditioned Prior Modeling (CPM) training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override the pretrained MPNet directory from the config.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--enable_tsne",
        action="store_true",
        help="Generate t-SNE visualizations during target-domain evaluation.",
    )
    return parser


def run_cpm_training(
    default_config_path: str,
) -> None:
    args = build_argument_parser(
        default_config_path
    ).parse_args()

    config = load_config(args.config)
    device = resolve_torch_device(config["gpu_id"])

    model_path = (
        args.model_path
        if args.model_path is not None
        else config["pretrained_model_path"]
    )
    run_count = (
        args.num_runs
        if args.num_runs is not None
        else config["num_experiment_runs"]
    )
    evaluation_interval = (
        args.eval_interval
        if args.eval_interval is not None
        else config["evaluation_interval"]
    )
    log_interval = (
        args.log_interval
        if args.log_interval is not None
        else config["log_interval"]
    )

    seeds = config["seeds"][:run_count]
    if len(seeds) < run_count:
        raise ValueError(
            f"Requested {run_count} runs but only "
            f"{len(config['seeds'])} seeds are configured."
        )

    set_reproducible_seed(0)

    source_meta_training_data = load_source_meta_training_data(
        config["dataset_root"],
        config["source_dataset_file"],
    )
    standardized_target_cube, target_ground_truth = (
        load_standardized_mat_dataset(
            image_file=os.path.join(
                config["dataset_root"],
                config["target_image_file"],
            ),
            ground_truth_file=os.path.join(
                config["dataset_root"],
                config["target_ground_truth_file"],
            ),
            image_mat_key=config["target_image_mat_key"],
            ground_truth_mat_key=config[
                "target_ground_truth_mat_key"
            ],
        )
    )

    experiment_name = (
        f"CPM_{config['dataset_slug']}_"
        f"{config['target_class_count']}way_"
        f"{config['target_labeled_samples_per_class']}"
        "labels_per_class"
    )
    experiment_log_directory = os.path.join(
        config["log_directory"],
        experiment_name,
    )
    configure_experiment_logging(
        experiment_log_directory,
        run_count,
    )
    logger = logging.getLogger("cpm")
    logger.info("method_name: CPM")
    logger.info("dataset: %s", config["dataset_display_name"])
    logger.info("device: %s", device)
    logger.info("seeds: %s", seeds)

    overall_accuracies = np.zeros((run_count, 1))
    per_class_accuracies = np.zeros(
        (run_count, config["target_class_count"])
    )
    kappa_scores = np.zeros((run_count, 1))

    final_map_state = None
    last_train_start = None
    last_train_end = None
    last_test_end = None

    for run_index, seed in enumerate(seeds):
        set_reproducible_seed(seed)
        logger.info(
            "Starting run %s/%s with seed %s",
            run_index + 1,
            run_count,
            seed,
        )

        (
            train_loader,
            test_loader,
            target_meta_training_data,
            map_ground_truth,
            train_test_permutation,
            row_indices,
            column_indices,
            train_count,
        ) = prepare_target_domain_data(
            standardized_cube=standardized_target_cube,
            ground_truth=target_ground_truth,
            class_count=config["target_class_count"],
            target_labeled_samples_per_class=config[
                "target_labeled_samples_per_class"
            ],
            patch_size=config["patch_size"],
        )

        models = build_cpm_models(
            config=config,
            device=device,
            model_path=model_path,
        )

        initialize_trainable_modules(
            device,
            models.source_spectral_mapping,
            models.target_spectral_mapping,
            models.visual_encoder,
            models.semantic_projection,
            models.prototype_refinement,
        )
        models.train()

        log_model_parameter_counts(
            logger,
            {
                "Source spectral mapping":
                    models.source_spectral_mapping,
                "Target spectral mapping":
                    models.target_spectral_mapping,
                "Spectral-spatial encoder":
                    models.visual_encoder,
                "CCSA prompt-tuned MPNet":
                    models.semantic_aggregation,
                "Category semantic projection":
                    models.semantic_projection,
                "Prototype refinement network":
                    models.prototype_refinement,
            },
        )

        optimizers = build_cpm_optimizers(
            models,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        cross_entropy_loss = nn.CrossEntropyLoss().to(device)
        cra_loss_function = CategoryRelationalAlignmentLoss(
            batch_size=config["target_class_count"],
            device=device,
            temperature=config["cra_temperature"],
        ).to(device)

        encoded_source_category_names = batch_to_device(
            models.semantic_aggregation.tokenize_category_names(
                config["source_category_names"]
            ),
            device,
        )
        encoded_target_category_names = batch_to_device(
            models.semantic_aggregation.tokenize_category_names(
                config["target_category_names"]
            ),
            device,
        )

        writer = SummaryWriter(
            log_dir=os.path.join(
                experiment_log_directory,
                f"run_{run_index}",
            )
        )

        best_accuracy = 0.0
        best_episode_index = 0
        cumulative_source_hits = 0.0
        cumulative_source_count = 0.0
        cumulative_target_hits = 0.0
        cumulative_target_count = 0.0

        train_start = time.time()
        last_train_start = train_start

        for episode_index in range(
            config["num_training_episodes"]
        ):
            source_task = EpisodicClassificationTask(
                source_meta_training_data,
                class_count=config["target_class_count"],
                support_samples_per_class=config[
                    "episode_support_samples_per_class"
                ],
                query_samples_per_class=config[
                    "episode_query_samples_per_class"
                ],
            )
            target_task = EpisodicClassificationTask(
                target_meta_training_data,
                class_count=config["target_class_count"],
                support_samples_per_class=config[
                    "episode_support_samples_per_class"
                ],
                query_samples_per_class=config[
                    "episode_query_samples_per_class"
                ],
            )

            total_loss, episode_statistics = (
                run_cross_domain_training_episode(
                    source_task=source_task,
                    target_task=target_task,
                    models=models,
                    cross_entropy_loss=cross_entropy_loss,
                    cra_loss_function=cra_loss_function,
                    encoded_source_category_names=(
                        encoded_source_category_names
                    ),
                    encoded_target_category_names=(
                        encoded_target_category_names
                    ),
                    config=config,
                    device=device,
                )
            )

            zero_optimizer_gradients(optimizers)
            total_loss.backward()
            step_optimizers(optimizers)

            cumulative_source_hits += episode_statistics[
                "source_hits"
            ]
            cumulative_source_count += episode_statistics[
                "source_count"
            ]
            cumulative_target_hits += episode_statistics[
                "target_hits"
            ]
            cumulative_target_count += episode_statistics[
                "target_count"
            ]

            source_accuracy = (
                cumulative_source_hits
                / cumulative_source_count
            )
            target_accuracy = (
                cumulative_target_hits
                / cumulative_target_count
            )

            if (episode_index + 1) % log_interval == 0:
                logger.info(
                    "episode=%d classification_loss=%.4f "
                    "cra_loss=%.4f total_loss=%.4f "
                    "source_acc=%.4f target_acc=%.4f",
                    episode_index + 1,
                    episode_statistics[
                        "classification_loss"
                    ].item(),
                    episode_statistics["cra_loss"].item(),
                    episode_statistics[
                        "total_loss"
                    ].item(),
                    source_accuracy,
                    target_accuracy,
                )
                writer.add_scalar(
                    "Loss/classification",
                    episode_statistics[
                        "classification_loss"
                    ].item(),
                    episode_index + 1,
                )
                writer.add_scalar(
                    "Loss/CRA",
                    episode_statistics["cra_loss"].item(),
                    episode_index + 1,
                )
                writer.add_scalar(
                    "Loss/total",
                    episode_statistics[
                        "total_loss"
                    ].item(),
                    episode_index + 1,
                )
                writer.add_scalar(
                    "Accuracy/source_episode_running",
                    source_accuracy,
                    episode_index + 1,
                )
                writer.add_scalar(
                    "Accuracy/target_episode_running",
                    target_accuracy,
                    episode_index + 1,
                )

            should_evaluate = (
                (episode_index + 1) % evaluation_interval == 0
                or episode_index == 0
            )
            if should_evaluate:
                last_train_end = time.time()
                tsne_output_path = os.path.join(
                    config["tsne_directory"],
                    config["dataset_slug"],
                    (
                        f"{config['dataset_slug']}_seed_{seed}_"
                        f"episode_{episode_index + 1}.png"
                    ),
                )

                with torch.no_grad():
                    (
                        test_accuracy,
                        test_labels,
                        test_predictions,
                    ) = evaluate_target_domain(
                        models=models,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        encoded_target_category_names=(
                            encoded_target_category_names
                        ),
                        device=device,
                        writer=writer,
                        episode_index=episode_index,
                        logger=logger,
                        enable_tsne=args.enable_tsne,
                        tsne_output_path=tsne_output_path,
                        class_colors=config["class_colors"],
                    )
                last_test_end = time.time()

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_episode_index = episode_index
                    overall_accuracies[run_index] = (
                        test_accuracy
                    )

                    confusion = metrics.confusion_matrix(
                        test_labels,
                        test_predictions,
                        labels=np.arange(
                            config["target_class_count"]
                        ),
                    )
                    class_totals = np.sum(
                        confusion,
                        axis=1,
                        dtype=float,
                    )
                    per_class_accuracies[run_index, :] = (
                        np.divide(
                            np.diag(confusion),
                            class_totals,
                            out=np.zeros_like(
                                class_totals,
                                dtype=float,
                            ),
                            where=class_totals != 0,
                        )
                    )

                    kappa_scores[run_index] = (
                        metrics.cohen_kappa_score(
                            test_labels,
                            test_predictions,
                        )
                    )

                    final_map_state = {
                        "predictions": test_predictions.copy(),
                        "ground_truth":
                            map_ground_truth.copy(),
                        "permutation":
                            train_test_permutation.copy(),
                        "row_indices": row_indices.copy(),
                        "column_indices":
                            column_indices.copy(),
                        "train_count": train_count,
                    }

                logger.info(
                    "best episode=%d, best accuracy=%.2f",
                    best_episode_index + 1,
                    best_accuracy,
                )

        writer.close()
        logger.info(
            "Run %d best episode=%d, best accuracy=%.2f",
            run_index,
            best_episode_index + 1,
            best_accuracy,
        )
        if last_train_end is not None:
            logger.info(
                "Train time for current run (s): %.5f",
                last_train_end - train_start,
            )

    overall_accuracy_mean = np.mean(overall_accuracies)
    overall_accuracy_std = np.std(overall_accuracies)

    average_accuracy_per_run = np.mean(
        per_class_accuracies,
        axis=1,
    )
    average_accuracy_mean = np.mean(
        average_accuracy_per_run
    )
    average_accuracy_std = np.std(
        average_accuracy_per_run
    )

    kappa_mean = np.mean(kappa_scores)
    kappa_std = np.std(kappa_scores)

    per_class_mean = np.mean(
        per_class_accuracies,
        axis=0,
    )
    per_class_std = np.std(
        per_class_accuracies,
        axis=0,
    )

    if (
        last_train_end is not None
        and last_train_start is not None
    ):
        logger.info(
            "Train time for final run (s): %.5f",
            last_train_end - last_train_start,
        )
    if (
        last_test_end is not None
        and last_train_end is not None
    ):
        logger.info(
            "Final evaluation time (s): %.5f",
            last_test_end - last_train_end,
        )

    logger.info(
        "average OA: %.2f +- %.2f",
        overall_accuracy_mean,
        overall_accuracy_std,
    )
    logger.info(
        "average AA: %.2f +- %.2f",
        100 * average_accuracy_mean,
        100 * average_accuracy_std,
    )
    logger.info(
        "average kappa: %.4f +- %.4f",
        100 * kappa_mean,
        100 * kappa_std,
    )
    logger.info("accuracy list: %s", overall_accuracies)

    for class_index in range(
        config["target_class_count"]
    ):
        logger.info(
            "Class %d: %.2f +- %.2f",
            class_index,
            100 * per_class_mean[class_index],
            100 * per_class_std[class_index],
        )

    if final_map_state is not None:
        classification_map_path = os.path.join(
            config["classification_map_directory"],
            (
                f"cpm_{config['dataset_slug']}_"
                f"{config['target_labeled_samples_per_class']}"
                "labels_per_class.png"
            ),
        )
        save_classification_map(
            predictions=final_map_state["predictions"],
            ground_truth=final_map_state["ground_truth"],
            permutation=final_map_state["permutation"],
            row_indices=final_map_state["row_indices"],
            column_indices=final_map_state[
                "column_indices"
            ],
            train_count=final_map_state["train_count"],
            patch_size=config["patch_size"],
            class_colors=config["class_colors"],
            output_path=classification_map_path,
        )
