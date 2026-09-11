# UI Copy Refactor Review

The desktop preview shows the revised hero title, concise subtitle, clearer model labels, plain-language training controls, readable comparison table headings, and a shorter methodology section. The original dark visual system remains intact.

The mobile preview shows the same copy hierarchy in a single-column flow with the labels remaining legible and the training, comparison, export, and methodology sections continuing to stack correctly. No interaction logic or measurement-state behavior was changed by the wording pass.

Verification completed: `pnpm run check`, Vitest, and `pnpm run build` passed after the copy changes. The live page continues to show explicit `Not measured` values when artifacts or the API are unavailable.

## Final verification

The final desktop preview keeps the revised hero and section hierarchy readable while exposing the Temperature, Alpha, F1 score, INT8, ONNX, and TorchScript explanations directly in the interface. The mobile preview preserves the same explanations in the stacked training and comparison sections. The dark visual system and explicit `Not measured` states remain unchanged.
