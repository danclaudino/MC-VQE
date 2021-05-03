/*******************************************************************************
 * Copyright (c) 2021 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Daniel Claudino - initial API and implementation
 ******************************************************************************/
#include "mc-vqe.hpp"
#include "importable/ImportFromTestFile.hpp"


#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"

#include <memory>

using namespace cppmicroservices;

namespace {

class US_ABI_LOCAL MC_VQE_Activator : public BundleActivator {

public:
  MC_VQE_Activator() {}

  void Start(BundleContext context) {
    auto mcvqe = std::make_shared<xacc::algorithm::MC_VQE>();
    context.RegisterService<xacc::Algorithm>(mcvqe);

    auto test = std::make_shared<xacc::ImportFromTestFile>();
    context.RegisterService<xacc::Importable>(test);

  }

  void Stop(BundleContext /*context*/) {}
};

}

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(MC_VQE_Activator)