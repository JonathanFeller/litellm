"use client";

import { ArrowLeft, Pencil, Trash2 } from "lucide-react";
import { useState } from "react";

import { DateCell } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

import { AutoRouterRow } from "./autoRouterRows";

interface AutoRouterDetailsPageProps {
  router: AutoRouterRow;
  canModify: boolean;
  onBack: () => void;
  onEdit: () => void;
  onDelete: () => void;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <div className="text-sm text-foreground">{children}</div>
    </div>
  );
}

export function AutoRouterDetailsPage({ router, canModify, onBack, onEdit, onDelete }: AutoRouterDetailsPageProps) {
  const [showAllTargets, setShowAllTargets] = useState(false);
  const visibleTargets = showAllTargets ? router.targets : router.targets.slice(0, 12);

  return (
    <div className="w-full space-y-4">
      <Button variant="ghost" size="sm" onClick={onBack}>
        <ArrowLeft />
        Back to auto routers
      </Button>

      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold text-foreground">{router.name}</h2>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{router.id}</p>
        </div>
        {canModify && (router.canEdit || router.canDelete) && (
          <div className="flex shrink-0 items-center gap-2">
            {router.canEdit && (
              <Button variant="secondary" onClick={onEdit}>
                <Pencil />
                Edit
              </Button>
            )}
            {router.canDelete && (
              <Button variant="destructive" onClick={onDelete}>
                <Trash2 />
                Delete
              </Button>
            )}
          </div>
        )}
      </div>

      {router.editBlockedReason && (
        <p className="text-sm text-muted-foreground">
          {router.editBlockedReason === "config-managed"
            ? "Defined in config.yaml, so it cannot be edited or deleted from the dashboard."
            : `${router.typeLabel} routers have no dashboard editor; change their settings in config.yaml. This one was created through the API, so it can still be deleted.`}
        </p>
      )}

      <Card className="p-6">
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Type">
            <Badge variant="secondary" className="font-normal">
              {router.typeLabel}
            </Badge>
          </Field>
          <Field label="Routing strategy">{router.strategy}</Field>
          <Field label="Default model">
            {router.defaultModel ? (
              <Badge variant="secondary" className="font-normal">
                {router.defaultModel}
              </Badge>
            ) : (
              <span className="text-muted-foreground">-</span>
            )}
          </Field>
          <Field label="Created">
            <DateCell value={router.createdAt} precision="date" />
          </Field>
        </div>

        <div className="mt-6 border-t border-border pt-6">
          <Field label={`Routes to (${router.targets.length})`}>
            {router.targets.length === 0 ? (
              <span className="text-muted-foreground">-</span>
            ) : (
              <div className="flex flex-wrap items-center gap-1.5">
                {visibleTargets.map((target) => (
                  <Badge key={target} variant="secondary" className="font-normal">
                    {target}
                  </Badge>
                ))}
                {!showAllTargets && router.targets.length > visibleTargets.length && (
                  <Button variant="ghost" size="sm" onClick={() => setShowAllTargets(true)}>
                    Show {router.targets.length - visibleTargets.length} more
                  </Button>
                )}
              </div>
            )}
          </Field>
        </div>
      </Card>
    </div>
  );
}
