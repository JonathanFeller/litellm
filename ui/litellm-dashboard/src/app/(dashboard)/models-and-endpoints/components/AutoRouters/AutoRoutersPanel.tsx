"use client";

import { ArrowLeft, Plus } from "lucide-react";
import { useMemo, useState } from "react";

import { useAutoRouters, useInvalidateAutoRouters } from "@/app/(dashboard)/hooks/models/useModels";
import AddAutoRouterTab from "@/components/add_model/add_auto_router_tab";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import EditAutoRouterModal from "@/components/edit_auto_router/edit_auto_router_modal";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { modelDeleteCall } from "@/components/networking";
import { Button } from "@/components/ui/button";

import { AutoRouterDetailsPage } from "./AutoRouterDetailsPage";
import { AutoRoutersTable } from "./AutoRoutersTable";
import { AutoRouterRow, toAutoRouterRows } from "./autoRouterRows";
import { useAutoRouterDetailRouting } from "./useAutoRouterDetailRouting";

interface AutoRoutersPanelProps {
  accessToken: string;
  userRole: string;
  /** Owned by the page, which knows whether the caller is a proxy admin or a team admin. */
  canModify: boolean;
}

export function AutoRoutersPanel({ accessToken, userRole, canModify }: AutoRoutersPanelProps) {
  const { data: deployments, isLoading } = useAutoRouters();
  const invalidateAutoRouters = useInvalidateAutoRouters();
  const [isCreating, setIsCreating] = useState(false);
  const [editingRouter, setEditingRouter] = useState<AutoRouterRow | null>(null);
  const [deletingRouter, setDeletingRouter] = useState<AutoRouterRow | null>(null);
  const { autoRouterId, openAutoRouter, close } = useAutoRouterDetailRouting();
  const [isDeleting, setIsDeleting] = useState(false);

  const routers = useMemo(() => toAutoRouterRows(deployments ?? []), [deployments]);
  const selectedRouter = autoRouterId ? routers.find((row) => row.id === autoRouterId) ?? null : null;

  const handleCreated = () => {
    setIsCreating(false);
    void invalidateAutoRouters();
  };

  const handleEdited = () => {
    setEditingRouter(null);
    void invalidateAutoRouters();
  };

  const handleConfirmDelete = async () => {
    if (!deletingRouter) return;
    setIsDeleting(true);
    try {
      await modelDeleteCall(accessToken, deletingRouter.id);
      NotificationsManager.success(`Deleted auto router: ${deletingRouter.name}`);
      const wasOpen = autoRouterId === deletingRouter.id;
      setDeletingRouter(null);
      // The detail view of a deleted router has nothing left to show.
      if (wasOpen) close();
      await invalidateAutoRouters();
    } catch (error) {
      NotificationsManager.fromBackend(`Failed to delete auto router: ${error}`);
    } finally {
      setIsDeleting(false);
    }
  };

  const renderDialogs = () => (
    <>
      {deletingRouter && (
        <DeleteResourceModal
          isOpen
          title="Delete Auto Router"
          message={`Are you sure you want to delete "${deletingRouter.name}"? Any client still calling this model name will start failing.`}
          resourceInformationTitle="Auto router"
          resourceInformation={[
            { label: "Name", value: deletingRouter.name },
            { label: "Type", value: deletingRouter.kind === "complexity" ? "Complexity" : "Semantic" },
            { label: "ID", value: deletingRouter.id },
          ]}
          onCancel={() => setDeletingRouter(null)}
          onOk={handleConfirmDelete}
          confirmLoading={isDeleting}
        />
      )}

      {editingRouter && canModify && (
        <EditAutoRouterModal
          isVisible
          onCancel={() => setEditingRouter(null)}
          onSuccess={handleEdited}
          modelData={editingRouter.deployment}
          accessToken={accessToken}
          userRole={userRole}
        />
      )}
    </>
  );

  if (selectedRouter) {
    return (
      <div className="w-full space-y-4">
        <AutoRouterDetailsPage
          router={selectedRouter}
          canModify={canModify}
          onBack={close}
          onEdit={() => setEditingRouter(selectedRouter)}
          onDelete={() => setDeletingRouter(selectedRouter)}
        />
        {renderDialogs()}
      </div>
    );
  }

  if (isCreating && canModify) {
    return (
      <div className="w-full space-y-4">
        <Button variant="ghost" size="sm" onClick={() => setIsCreating(false)}>
          <ArrowLeft />
          Back to auto routers
        </Button>
        <AddAutoRouterTab handleOk={handleCreated} accessToken={accessToken} userRole={userRole} />
      </div>
    );
  }

  return (
    <div className="w-full space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-foreground">Auto routers</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Auto routers sit above your deployments and pick a model per request. They are called like any other model,
            so clients keep using a single model name.
          </p>
        </div>
        {canModify && (
          <Button onClick={() => setIsCreating(true)} className="shrink-0">
            <Plus />
            Add Auto Router
          </Button>
        )}
      </div>

      <AutoRoutersTable
        routers={routers}
        isLoading={isLoading}
        canModify={canModify}
        onRouterClick={(row) => openAutoRouter(row.id)}
        onDeleteClick={setDeletingRouter}
      />

      {renderDialogs()}
    </div>
  );
}
