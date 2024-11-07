import torch
from transformers import PatchTSMixerForTimeSeriesClassification, PatchTSMixerForPretraining
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple

@dataclass
class PatchTSMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input data to the model.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*):
            Bool Tensor indicating True in masked patches and False otherwise.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patch_input: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@dataclass
class CustomPatchTSMixerForTimeSeriesClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForTimeSeriesClassificationOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Prediction output from the classfication head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    labels: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class CustomPatchTSMixerForTimeSeriesClassification(PatchTSMixerForTimeSeriesClassification):
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> CustomPatchTSMixerForTimeSeriesClassificationOutput:
        r"""
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """

        loss = torch.nn.CrossEntropyLoss()

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        model_output = self.model(
            past_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # x: [batch_size x nvars x num_patch x d_model]
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state,
                loc=model_output.loc,
                scale=model_output.scale,
            )  # x: [batch_size x nvars x num_patch x d_model]

        y_hat = self.head(model_output.last_hidden_state)  # tensor [batch_size x n_labels]

        if target_values is not None and return_loss is True:
            loss_val = loss(y_hat, target_values)
        else:
            loss_val = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    target_values,
                    y_hat,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                ]
            )

        return CustomPatchTSMixerForTimeSeriesClassificationOutput(
            loss=loss_val,
            prediction_outputs=y_hat,  # tensor [batch_size x n_labels]
            labels = target_values,
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )
    
@dataclass
class CustomPatchTSMixerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForPreTrainingOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    true_values: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class CustomPatchTSMixerForPretraining(PatchTSMixerForPretraining):
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> CustomPatchTSMixerForPreTrainingOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if self.masked_loss is True:
            loss = torch.nn.MSELoss(reduction="none")
        else:
            loss = torch.nn.MSELoss(reduction="mean")

        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # x.last_hidden_state: [batch_size x nvars x num_patch x d_model]
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        x_hat = self.head(model_output.last_hidden_state)  # tensor [batch_size x nvars x num_patch x patch_length]

        if return_loss is True:
            loss_val = loss(x_hat, model_output.patch_input)
        else:
            loss_val = None

        # calculate masked_loss
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    x_hat,
                    model_output.patch_input,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                ]
            )

        return CustomPatchTSMixerForPreTrainingOutput(
            loss=loss_val,
            prediction_outputs=x_hat,  # tensor [batch_size x nvars x num_patch x patch_length]
            true_values=model_output.patch_input,
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )
