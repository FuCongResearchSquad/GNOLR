from model.DSSM import DSSM
from model.NSB import NSB
from model.ESMM import ESMM
from model.IPW import IPW
from model.DR import DR
from model.DCMT import DCMT
from model.NISE import NISE
from model.TAFE import TAFE


def get_model(
    model,
    user_feature,
    item_feature,
    user_dnn_size,
    item_dnn_size,
    dropout,
    activation,
    use_senet,
    dimensions,
    l2_normalization=False,
    output=1,
    similarity="dot",
    loss="bceloss",
    tower="base",
):
    if isinstance(model, str):
        model = model.lower()
        if model == "dssm":
            return DSSM(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "nsb":
            return NSB(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "esmm":
            return ESMM(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "ipw":
            return IPW(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "dr":
            return DR(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "dcmt":
            return DCMT(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "nise":
            return NISE(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "tafe":
            return TAFE(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "nolr":
            return DSSM(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
        elif model == "gnolr":
            return DSSM(
                user_feature,
                item_feature,
                user_dnn_size=user_dnn_size,
                item_dnn_size=item_dnn_size,
                l2_normalization=l2_normalization,
                loss=loss,
                dropout=dropout,
                activation=activation,
                use_senet=use_senet,
                dimensions=dimensions,
                output=output,
                similarity=similarity,
                tower=tower,
            )
